#!/usr/bin/env python3
"""
Tool Orchestrator Interactive CLI

A command-line interface for testing LLM orchestration
with tools and delegate LLMs.
"""

import argparse
import atexit
import json
import logging
import signal
import sys
import threading

from .config import config
from .orchestrator import ToolOrchestrator
from .llm_call import LLMClient

# Global shutdown flag for signal handling
_shutdown_requested = threading.Event()
_active_clients: list = []  # Track OpenAI clients for cleanup

logger = logging.getLogger(__name__)


def _signal_handler(signum: int, frame) -> None:
    """Handle SIGINT for graceful shutdown."""
    if _shutdown_requested.is_set():
        # Second interrupt - force exit
        logger.debug("Force shutdown requested")
        _cleanup_clients()
        sys.exit(1)
    else:
        # First interrupt - request graceful shutdown
        logger.debug("Shutdown requested")
        _shutdown_requested.set()
        print("\n\nShutting down... (press Ctrl+C again to force)")


def _cleanup_clients() -> None:
    """Close all tracked OpenAI clients."""
    for client in _active_clients:
        try:
            if hasattr(client, "close") and not getattr(client, "is_closed", False):
                client.close()
                logger.debug(f"Closed client: {type(client).__name__}")
        except Exception as e:
            logger.debug(f"Error closing client: {e}")
    _active_clients.clear()


def _register_client(client) -> None:
    """Register an OpenAI client for cleanup on exit."""
    if hasattr(client, "close"):
        _active_clients.append(client)


# Register cleanup on normal exit
atexit.register(_cleanup_clients)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def print_banner() -> None:
    """Print the welcome banner."""
    banner = """
╔════════════════════════════════════════════════════════════════╗
║                  Tool Orchestrator Interactive                  ║
║                                                                 ║
║  ReAct-style LLM orchestration with tools and delegates        ║
╚════════════════════════════════════════════════════════════════╝

Available commands:
  /help     - Show this help message
  /trace    - Show the trace of the last query
  /tools    - List available tools
  /verbose  - Toggle verbose mode
  /clear    - Clear conversation history
  /quit     - Exit the CLI

Type your questions or tasks below.
"""
    print(banner)


def print_tools() -> None:
    """Print available tools."""
    print(
        """
Available Tools:
────────────────────────────────────────────────────────────────
1. web_search      - Search the web via SearXNG
2. python_execute  - Execute Python code in a sandbox
3. calculate       - Evaluate mathematical expressions

Delegate LLMs:
────────────────────────────────────────────────────────────────"""
    )

    # Dynamically load delegate tools from config
    for i, (role, delegate) in enumerate(config.delegates.items(), start=4):
        tool_name = delegate.tool_name.ljust(16)
        print(f"{i}. {tool_name} - {delegate.display_name}")
    print()


def print_trace(orchestrator: ToolOrchestrator) -> None:
    """Print the trace of the last orchestration run."""
    trace = orchestrator.get_trace()
    if not trace:
        print("\nNo trace available. Run a query first.\n")
        return

    print("\n" + "═" * 70)
    print("ORCHESTRATION TRACE")
    print("═" * 70)

    for step in trace:
        print(f"\n┌─ Step {step['step']}" + ("  [FINAL]" if step["is_final"] else ""))
        print("│")
        if step["reasoning"]:
            print(f"│  Thought: {step['reasoning']}")
        if step["action"]:
            print(f"│  Action: {step['action']}")
        if step["action_input"]:
            if isinstance(step["action_input"], dict):
                print(f"│  Input: {json.dumps(step['action_input'], indent=2)}")
            else:
                print(f"│  Input: {step['action_input']}")
        if step["observation"]:
            obs = step["observation"]
            if len(obs) > 200:
                obs = obs[:200] + "..."
            print(f"│  Observation: {obs}")
        if step["final_answer"]:
            print(f"│  Final Answer: {step['final_answer']}")
        print("└" + "─" * 68)

    print()


class InteractiveCLI:
    """Interactive CLI for Tool Orchestrator."""

    def __init__(self, verbose: bool = False):
        """Initialize the CLI."""
        self.verbose = verbose
        self.llm_client = LLMClient()
        self.orchestrator = ToolOrchestrator(
            llm_client=self.llm_client,
            verbose=verbose,
        )
        # Register the OpenAI client for cleanup
        if hasattr(self.llm_client, "orchestrator_client"):
            _register_client(self.llm_client.orchestrator_client)

    def toggle_verbose(self) -> None:
        """Toggle verbose mode."""
        self.verbose = not self.verbose
        self.orchestrator.verbose = self.verbose
        print(f"\nVerbose mode: {'ON' if self.verbose else 'OFF'}\n")

    def clear_history(self) -> None:
        """Clear the orchestration history."""
        self.orchestrator.steps = []
        print("\nConversation history cleared.\n")

    def process_query(self, query: str) -> bool:
        """Process a user query.

        Returns:
            True if should continue, False if shutdown requested
        """
        print("\n" + "─" * 70)
        print("Processing query...")
        print("─" * 70 + "\n")

        try:
            result = self.orchestrator.run(query)

            # Check if shutdown was requested during query
            if _shutdown_requested.is_set():
                print("\n\nQuery completed, shutting down.\n")
                return False

            print("\n" + "═" * 70)
            print("ANSWER")
            print("═" * 70)
            print(result)
            print("═" * 70 + "\n")

            # Show step count
            step_count = len(self.orchestrator.steps)
            print(f"(Completed in {step_count} step{'s' if step_count != 1 else ''})")
            print("Use /trace to see the full reasoning trace.\n")

        except KeyboardInterrupt:
            _shutdown_requested.set()
            print("\n\nQuery interrupted, shutting down.\n")
            return False
        except Exception as e:
            # Check if this is a shutdown-related error
            if _shutdown_requested.is_set() or "shutdown" in str(e).lower():
                print("\n\nShutdown in progress.\n")
                return False
            print(f"\nError: {e}\n")
            if self.verbose:
                import traceback

                traceback.print_exc()

        return True

    def run(self) -> None:
        """Run the interactive CLI loop."""
        print_banner()

        while not _shutdown_requested.is_set():
            try:
                # Get user input
                user_input = input(">>> ").strip()

                if _shutdown_requested.is_set():
                    break

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    command = user_input.lower()

                    if command in ("/quit", "/exit", "/q"):
                        print("\nGoodbye!\n")
                        break
                    elif command in ("/help", "/h", "/?"):
                        print_banner()
                    elif command == "/trace":
                        print_trace(self.orchestrator)
                    elif command == "/tools":
                        print_tools()
                    elif command == "/verbose":
                        self.toggle_verbose()
                    elif command == "/clear":
                        self.clear_history()
                    else:
                        print(f"\nUnknown command: {user_input}")
                        print("Type /help for available commands.\n")
                else:
                    # Process as a query
                    if not self.process_query(user_input):
                        break

            except KeyboardInterrupt:
                if _shutdown_requested.is_set():
                    print("\n")
                    break
                print("\n\nType /quit to exit.\n")
            except EOFError:
                print("\nGoodbye!\n")
                break

        # Cleanup on exit
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        _cleanup_clients()


def main() -> None:
    """Main entry point."""
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)

    parser = argparse.ArgumentParser(
        description="Tool Orchestrator Interactive CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Start interactive mode
  %(prog)s -v                 # Start with verbose logging
  %(prog)s -q "What is 2+2?"  # Run a single query

Use /tools in interactive mode to see available tools and delegates.
""",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "-q",
        "--query",
        type=str,
        help="Run a single query and exit",
    )

    parser.add_argument(
        "--orchestrator-url",
        type=str,
        default=None,
        help=f"Orchestrator model endpoint URL (default: from ORCHESTRATOR_BASE_URL env or {config.orchestrator.base_url})",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (for scripting)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Create LLM client with custom URL if provided
    llm_client = LLMClient(orchestrator_url=args.orchestrator_url)
    _register_client(llm_client.orchestrator_client)

    try:
        if args.query:
            # Single query mode
            orchestrator = ToolOrchestrator(
                llm_client=llm_client,
                verbose=args.verbose,
            )

            result = orchestrator.run(args.query)

            if args.json:
                output = {
                    "query": args.query,
                    "answer": result,
                    "trace": orchestrator.get_trace(),
                }
                print(json.dumps(output, indent=2))
            else:
                print(result)
        else:
            # Interactive mode
            cli = InteractiveCLI(verbose=args.verbose)
            cli.llm_client = llm_client
            cli.orchestrator.llm_client = llm_client
            _register_client(cli.llm_client.orchestrator_client)
            cli.run()
    finally:
        _cleanup_clients()


if __name__ == "__main__":
    main()
