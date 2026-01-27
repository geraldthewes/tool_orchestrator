#!/usr/bin/env python
"""
Analyze Langfuse traces to diagnose DSPy optimization performance.

This script queries Langfuse traces and analyzes execution patterns to help
diagnose why DSPy optimization improvements may be slow.

Since scores (orchestration_quality_with_tools) are computed locally by metric
functions and NOT stored in Langfuse, this script analyzes execution quality
proxies: error rates, tool usage patterns, latency, and token consumption.

Usage:
    python scripts/analyze_langfuse_traces.py --limit 50 --verbose
    python scripts/analyze_langfuse_traces.py --trace-name dspy_optimization --output report.json
    python scripts/analyze_langfuse_traces.py --since 2024-01-01T00:00:00
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langfuse import Langfuse

from src.config import config

logger = logging.getLogger(__name__)


def create_client(host: str, public_key: str, secret_key: str) -> Langfuse:
    """Create Langfuse client with given credentials."""
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )


def fetch_traces(
    client: Langfuse,
    name: str | None,
    limit: int,
    since: str | None = None,
) -> list:
    """
    Fetch traces from Langfuse API with pagination.

    Args:
        client: Langfuse client instance
        name: Filter traces by name (None to fetch all traces)
        limit: Maximum number of traces to fetch
        since: Optional ISO timestamp to filter traces after this time

    Returns:
        List of trace objects
    """
    all_traces = []
    page = 1

    while len(all_traces) < limit:
        batch_size = min(50, limit - len(all_traces))
        try:
            # Build kwargs conditionally to avoid passing None
            kwargs = {"limit": batch_size, "page": page}
            if name is not None:
                kwargs["name"] = name
            traces = client.api.trace.list(**kwargs)
        except Exception as e:
            logger.error(f"Failed to fetch traces (page {page}): {e}")
            break

        if not traces.data:
            break

        # Filter by timestamp if specified
        if since:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            traces_data = [
                t for t in traces.data if t.timestamp and t.timestamp >= since_dt
            ]
        else:
            traces_data = traces.data

        all_traces.extend(traces_data)

        if len(traces.data) < batch_size:
            break
        page += 1

    return all_traces[:limit]


def fetch_observations_for_trace(client: Langfuse, trace_id: str) -> list:
    """
    Fetch all observations (spans, generations) for a trace with pagination.

    Args:
        client: Langfuse client instance
        trace_id: The trace ID to fetch observations for

    Returns:
        List of observation objects
    """
    all_observations = []
    page = 1
    page_size = 100  # Langfuse API limit

    try:
        while True:
            observations = client.api.observations.get_many(
                trace_id=trace_id,
                limit=page_size,
                page=page,
            )
            if not observations.data:
                break
            all_observations.extend(observations.data)
            if len(observations.data) < page_size:
                break
            page += 1
        return all_observations
    except Exception as e:
        logger.warning(f"Failed to fetch observations for trace {trace_id}: {e}")
        return all_observations  # Return what we got so far


def analyze_generation(gen: Any) -> dict:
    """
    Analyze a single LLM generation observation.

    Args:
        gen: Generation observation object

    Returns:
        Dictionary with analysis results
    """
    result = {
        "name": gen.name,
        "model": getattr(gen, "model", None),
        "has_output": bool(gen.output),
        "has_error": getattr(gen, "status_message", None) is not None
        or getattr(gen, "level", None) == "ERROR",
        "duration_ms": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }

    # Calculate duration
    if gen.end_time and gen.start_time:
        duration = gen.end_time - gen.start_time
        result["duration_ms"] = duration.total_seconds() * 1000

    # Extract token usage - handle both dict and object-style usage
    if hasattr(gen, "usage") and gen.usage:
        usage = gen.usage
        if isinstance(usage, dict):
            result["prompt_tokens"] = usage.get("promptTokens") or usage.get(
                "prompt_tokens"
            )
            result["completion_tokens"] = usage.get("completionTokens") or usage.get(
                "completion_tokens"
            )
            result["total_tokens"] = usage.get("totalTokens") or usage.get(
                "total_tokens"
            )
        else:
            # Object-style usage (Langfuse SDK models)
            result["prompt_tokens"] = getattr(usage, "input", None)
            result["completion_tokens"] = getattr(usage, "output", None)
            result["total_tokens"] = getattr(usage, "total", None)

    return result


def analyze_tool_span(span: Any) -> dict:
    """
    Analyze a single tool execution span.

    Args:
        span: Span observation object

    Returns:
        Dictionary with analysis results
    """
    result = {
        "name": span.name,
        "has_output": bool(span.output),
        "has_error": getattr(span, "level", None) == "ERROR"
        or getattr(span, "status_message", None) is not None,
        "error_message": None,
        "duration_ms": None,
    }

    # Check for error in output
    if result["has_error"] and span.output:
        result["error_message"] = str(span.output)[:500]  # Truncate long errors

    # Calculate duration
    if span.end_time and span.start_time:
        duration = span.end_time - span.start_time
        result["duration_ms"] = duration.total_seconds() * 1000

    return result


def compute_statistics(
    traces: list,
    observations_by_trace: dict[str, list],
) -> dict:
    """
    Compute aggregate statistics from traces and observations.

    Args:
        traces: List of trace objects
        observations_by_trace: Dictionary mapping trace IDs to observation lists

    Returns:
        Dictionary with computed statistics
    """
    stats = {
        "total_traces": len(traces),
        "total_generations": 0,
        "total_tool_calls": 0,
        "generations_with_error": 0,
        "generations_without_output": 0,
        "tool_calls_with_error": 0,
        "tool_usage": defaultdict(int),
        "model_usage": defaultdict(int),
        "generation_latencies_ms": [],
        "tool_latencies_ms": [],
        "token_usage": {"prompt": [], "completion": [], "total": []},
        "traces_with_errors": 0,
        "error_messages": [],
    }

    for trace_id, observations in observations_by_trace.items():
        trace_has_error = False

        for obs in observations:
            obs_type = getattr(obs, "type", None)
            obs_name = getattr(obs, "name", "")
            logger.debug(
                f"Observation: type={obs_type}, name={obs_name}, "
                f"class={type(obs).__name__}"
            )

            if obs_type == "GENERATION":
                stats["total_generations"] += 1
                gen_info = analyze_generation(obs)

                if gen_info["has_error"]:
                    stats["generations_with_error"] += 1
                    trace_has_error = True
                if not gen_info["has_output"]:
                    stats["generations_without_output"] += 1

                if gen_info["duration_ms"] is not None:
                    stats["generation_latencies_ms"].append(gen_info["duration_ms"])
                if gen_info["prompt_tokens"] is not None:
                    stats["token_usage"]["prompt"].append(gen_info["prompt_tokens"])
                if gen_info["completion_tokens"] is not None:
                    stats["token_usage"]["completion"].append(
                        gen_info["completion_tokens"]
                    )
                if gen_info["total_tokens"] is not None:
                    stats["token_usage"]["total"].append(gen_info["total_tokens"])
                if gen_info["model"]:
                    stats["model_usage"][gen_info["model"]] += 1

            elif obs_type == "SPAN":
                obs_name = getattr(obs, "name", "")
                if obs_name and obs_name.startswith("tool:"):
                    stats["total_tool_calls"] += 1
                    tool_name = obs_name.replace("tool:", "")
                    stats["tool_usage"][tool_name] += 1
                    span_info = analyze_tool_span(obs)

                    if span_info["has_error"]:
                        stats["tool_calls_with_error"] += 1
                        trace_has_error = True
                        if span_info["error_message"]:
                            stats["error_messages"].append(
                                {
                                    "type": "tool",
                                    "name": tool_name,
                                    "message": span_info["error_message"],
                                }
                            )
                    if span_info["duration_ms"] is not None:
                        stats["tool_latencies_ms"].append(span_info["duration_ms"])

        if trace_has_error:
            stats["traces_with_errors"] += 1

    # Compute summary statistics
    stats["avg_generation_latency_ms"] = (
        sum(stats["generation_latencies_ms"]) / len(stats["generation_latencies_ms"])
        if stats["generation_latencies_ms"]
        else 0
    )
    stats["avg_tool_latency_ms"] = (
        sum(stats["tool_latencies_ms"]) / len(stats["tool_latencies_ms"])
        if stats["tool_latencies_ms"]
        else 0
    )

    # Latency percentiles
    if stats["generation_latencies_ms"]:
        sorted_gen = sorted(stats["generation_latencies_ms"])
        stats["p50_generation_latency_ms"] = sorted_gen[len(sorted_gen) // 2]
        stats["p95_generation_latency_ms"] = sorted_gen[int(len(sorted_gen) * 0.95)]
        stats["max_generation_latency_ms"] = sorted_gen[-1]

    if stats["tool_latencies_ms"]:
        sorted_tool = sorted(stats["tool_latencies_ms"])
        stats["p50_tool_latency_ms"] = sorted_tool[len(sorted_tool) // 2]
        stats["p95_tool_latency_ms"] = sorted_tool[int(len(sorted_tool) * 0.95)]
        stats["max_tool_latency_ms"] = sorted_tool[-1]

    # Token usage summary
    if stats["token_usage"]["prompt"]:
        stats["avg_prompt_tokens"] = sum(stats["token_usage"]["prompt"]) / len(
            stats["token_usage"]["prompt"]
        )
        stats["max_prompt_tokens"] = max(stats["token_usage"]["prompt"])
    if stats["token_usage"]["completion"]:
        stats["avg_completion_tokens"] = sum(stats["token_usage"]["completion"]) / len(
            stats["token_usage"]["completion"]
        )
        stats["max_completion_tokens"] = max(stats["token_usage"]["completion"])

    # Convert defaultdicts to regular dicts for JSON serialization
    stats["tool_usage"] = dict(stats["tool_usage"])
    stats["model_usage"] = dict(stats["model_usage"])

    return stats


def extract_failures(
    traces: list,
    observations_by_trace: dict[str, list],
) -> list[dict]:
    """
    Extract detailed information about failed evaluations.

    Args:
        traces: List of trace objects
        observations_by_trace: Dictionary mapping trace IDs to observation lists

    Returns:
        List of failure records with detailed information
    """
    failures = []

    for trace in traces:
        trace_failures = []
        observations = observations_by_trace.get(trace.id, [])

        for obs in observations:
            is_error = (
                getattr(obs, "level", None) == "ERROR"
                or getattr(obs, "status_message", None) is not None
            )

            if is_error:
                failure_info = {
                    "type": getattr(obs, "type", "UNKNOWN"),
                    "name": getattr(obs, "name", "unknown"),
                    "input": _truncate(str(obs.input), 500) if obs.input else None,
                    "output": _truncate(str(obs.output), 500) if obs.output else None,
                    "status_message": getattr(obs, "status_message", None),
                }
                trace_failures.append(failure_info)

        if trace_failures:
            failures.append(
                {
                    "trace_id": trace.id,
                    "trace_name": trace.name,
                    "timestamp": (
                        trace.timestamp.isoformat() if trace.timestamp else None
                    ),
                    "failures": trace_failures,
                }
            )

    return failures


def _truncate(text: str, max_length: int) -> str:
    """Truncate text to max length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def print_summary(stats: dict) -> None:
    """Print summary statistics to console in a readable format."""
    print("\n" + "=" * 70)
    print("LANGFUSE TRACE ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nTraces analyzed: {stats['total_traces']}")
    print(f"Traces with errors: {stats['traces_with_errors']}")
    print(f"Total LLM generations: {stats['total_generations']}")
    print(f"Total tool calls: {stats['total_tool_calls']}")

    # Error rates
    print("\n--- Error Rates ---")
    if stats["total_generations"] > 0:
        gen_error_rate = (
            stats["generations_with_error"] / stats["total_generations"] * 100
        )
        gen_no_output_rate = (
            stats["generations_without_output"] / stats["total_generations"] * 100
        )
        print(
            f"LLM generations with error: {stats['generations_with_error']} ({gen_error_rate:.1f}%)"
        )
        print(
            f"LLM generations without output: {stats['generations_without_output']} ({gen_no_output_rate:.1f}%)"
        )
    else:
        print("No LLM generations found")

    if stats["total_tool_calls"] > 0:
        tool_error_rate = (
            stats["tool_calls_with_error"] / stats["total_tool_calls"] * 100
        )
        print(
            f"Tool calls with error: {stats['tool_calls_with_error']} ({tool_error_rate:.1f}%)"
        )
    else:
        print("No tool calls found")

    # Tool usage distribution
    print("\n--- Tool Usage Distribution ---")
    if stats["tool_usage"]:
        for tool, count in sorted(stats["tool_usage"].items(), key=lambda x: -x[1]):
            pct = (
                count / stats["total_tool_calls"] * 100
                if stats["total_tool_calls"] > 0
                else 0
            )
            print(f"  {tool}: {count} ({pct:.1f}%)")
    else:
        print("  No tools used")

    # Model usage
    print("\n--- Model Usage ---")
    if stats["model_usage"]:
        for model, count in sorted(stats["model_usage"].items(), key=lambda x: -x[1]):
            print(f"  {model}: {count}")
    else:
        print("  No model information available")

    # Latency statistics
    print("\n--- Latency (ms) ---")
    print(f"LLM generation avg: {stats['avg_generation_latency_ms']:.0f}")
    if "p50_generation_latency_ms" in stats:
        print(
            f"  p50: {stats['p50_generation_latency_ms']:.0f}, "
            f"p95: {stats['p95_generation_latency_ms']:.0f}, "
            f"max: {stats['max_generation_latency_ms']:.0f}"
        )

    print(f"Tool execution avg: {stats['avg_tool_latency_ms']:.0f}")
    if "p50_tool_latency_ms" in stats:
        print(
            f"  p50: {stats['p50_tool_latency_ms']:.0f}, "
            f"p95: {stats['p95_tool_latency_ms']:.0f}, "
            f"max: {stats['max_tool_latency_ms']:.0f}"
        )

    # Token usage
    print("\n--- Token Usage ---")
    if "avg_prompt_tokens" in stats:
        print(
            f"Prompt tokens avg: {stats['avg_prompt_tokens']:.0f}, max: {stats['max_prompt_tokens']}"
        )
    if "avg_completion_tokens" in stats:
        print(
            f"Completion tokens avg: {stats['avg_completion_tokens']:.0f}, max: {stats['max_completion_tokens']}"
        )

    # Sample error messages
    if stats.get("error_messages"):
        print("\n--- Sample Error Messages (up to 5) ---")
        for err in stats["error_messages"][:5]:
            print(f"  [{err['type']}:{err['name']}] {err['message'][:100]}")

    print("\n" + "=" * 70)


def print_failures(failures: list, verbose: bool = False) -> None:
    """Print failure details to console."""
    if not failures:
        print("\nNo failures found in traces.")
        return

    print(f"\n--- Failure Details ({len(failures)} traces with failures) ---")

    for i, failure in enumerate(failures[:10]):  # Show first 10
        print(f"\nTrace {i + 1}: {failure['trace_id'][:16]}...")
        print(f"  Name: {failure['trace_name']}")
        print(f"  Timestamp: {failure['timestamp']}")
        print(f"  Failures: {len(failure['failures'])}")

        if verbose:
            for f in failure["failures"][:3]:  # Show first 3 failures per trace
                print(f"    - [{f['type']}:{f['name']}]")
                if f["status_message"]:
                    print(f"      Status: {f['status_message'][:100]}")
                if f["output"]:
                    print(f"      Output: {f['output'][:100]}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Langfuse traces for DSPy optimization diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis of recent traces
    python scripts/analyze_langfuse_traces.py --limit 50

    # Verbose output with failure details
    python scripts/analyze_langfuse_traces.py --limit 100 --verbose

    # Export detailed report to JSON
    python scripts/analyze_langfuse_traces.py --output report.json

    # Filter by timestamp
    python scripts/analyze_langfuse_traces.py --since 2024-01-15T00:00:00
        """,
    )

    parser.add_argument(
        "--host",
        default=config.langfuse.host,
        help=f"Langfuse host URL (default: {config.langfuse.host})",
    )
    parser.add_argument(
        "--public-key",
        default=config.langfuse.public_key,
        help="Langfuse public key (default: from config)",
    )
    parser.add_argument(
        "--secret-key",
        default=config.langfuse.secret_key,
        help="Langfuse secret key (default: from config)",
    )
    parser.add_argument(
        "--trace-name",
        default="dspy_optimization",
        help="Filter by trace name (default: dspy_optimization)",
    )
    parser.add_argument(
        "--all-traces",
        action="store_true",
        help="Analyze all traces regardless of name (ignores --trace-name)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of traces to fetch (default: 100)",
    )
    parser.add_argument(
        "--since",
        help="Only include traces after this ISO timestamp (e.g., 2024-01-15T00:00:00)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for detailed JSON report",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show individual trace and failure details",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Validate credentials
    if not args.public_key or not args.secret_key:
        logger.error(
            "Langfuse credentials not configured. "
            "Set public_key and secret_key in config or via --public-key/--secret-key"
        )
        return 1

    # Connect to Langfuse
    logger.info(f"Connecting to Langfuse at {args.host}")
    try:
        client = create_client(args.host, args.public_key, args.secret_key)
    except Exception as e:
        logger.error(f"Failed to create Langfuse client: {e}")
        return 1

    # Fetch traces
    trace_name = None if args.all_traces else args.trace_name
    name_filter = "all" if args.all_traces else args.trace_name
    logger.info(f"Fetching traces (name={name_filter}, limit={args.limit})")
    traces = fetch_traces(client, trace_name, args.limit, args.since)
    logger.info(f"Found {len(traces)} traces")

    if not traces:
        logger.warning("No traces found matching criteria")
        return 0

    # Fetch observations for each trace
    logger.info("Fetching observations for each trace...")
    observations_by_trace: dict[str, list] = {}
    for i, trace in enumerate(traces):
        if args.verbose:
            logger.debug(
                f"Fetching observations for trace {i + 1}/{len(traces)}: {trace.id}"
            )
        observations_by_trace[trace.id] = fetch_observations_for_trace(client, trace.id)

    total_observations = sum(len(obs) for obs in observations_by_trace.values())
    logger.info(
        f"Fetched {total_observations} observations across {len(traces)} traces"
    )

    # Compute statistics
    stats = compute_statistics(traces, observations_by_trace)
    print_summary(stats)

    # Extract and show failures
    failures = extract_failures(traces, observations_by_trace)
    print_failures(failures, verbose=args.verbose)

    # Save detailed report if requested
    if args.output:
        # Clean up stats for JSON serialization (remove raw lists)
        json_stats = {
            k: v
            for k, v in stats.items()
            if k not in ("generation_latencies_ms", "tool_latencies_ms", "token_usage")
        }
        json_stats["token_usage_summary"] = {
            "prompt_count": len(stats["token_usage"]["prompt"]),
            "completion_count": len(stats["token_usage"]["completion"]),
        }

        report = {
            "generated_at": datetime.now().isoformat(),
            "parameters": {
                "host": args.host,
                "trace_name": args.trace_name,
                "limit": args.limit,
                "since": args.since,
            },
            "statistics": json_stats,
            "failures": failures,
        }

        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Detailed report saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
