"""
SearXNG Web Search Tool

Provides web search capabilities via a SearXNG instance.
"""

import logging
from typing import Optional
import requests

from ..config import config

logger = logging.getLogger(__name__)


def search(
    query: str,
    categories: Optional[str] = None,
    num_results: int = 5,
) -> dict:
    """
    Search the web using SearXNG.

    Args:
        query: The search query
        categories: Optional category filter (e.g., "general", "images", "news")
        num_results: Maximum number of results to return

    Returns:
        Dictionary with search results
    """
    # Validate query is not empty
    if not query or not query.strip():
        return {
            "query": query,
            "error": 'Search query is empty. Please provide a search query in format: {"query": "your search terms"}',
            "results": [],
            "total": 0,
        }

    params = {
        "q": query,
        "format": "json",
    }
    if categories:
        params["categories"] = categories

    try:
        response = requests.get(
            config.tools.searxng.url,
            params=params,
            timeout=config.tools.searxng.timeout,
        )
        response.raise_for_status()
        data = response.json()

        # Extract and format results
        results = []
        for result in data.get("results", [])[:num_results]:
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "engine": result.get("engine", ""),
            })

        return {
            "query": query,
            "results": results,
            "total": len(results),
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Search failed: {e}")
        return {
            "query": query,
            "error": str(e),
            "results": [],
            "total": 0,
        }


def format_results_for_llm(search_results: dict) -> str:
    """
    Format search results into a string suitable for LLM consumption.

    Args:
        search_results: Results from search()

    Returns:
        Formatted string of search results
    """
    if search_results.get("error"):
        return f"Search error: {search_results['error']}"

    if not search_results.get("results"):
        return "No results found."

    formatted = f"Search results for '{search_results['query']}':\n\n"
    for i, result in enumerate(search_results["results"], 1):
        formatted += f"{i}. {result['title']}\n"
        formatted += f"   URL: {result['url']}\n"
        if result['content']:
            formatted += f"   {result['content'][:200]}...\n"
        formatted += "\n"

    return formatted


def _handle_search(params: dict) -> dict:
    """Handle search tool invocation with input validation."""
    if "raw" in params and "query" not in params:
        return {
            "query": None,
            "error": 'Invalid input format. Expected JSON: {"query": "your search terms"}. Received unparseable input.',
            "results": [],
            "total": 0,
        }
    return search(
        query=params.get("query", ""),
        categories=params.get("categories"),
        num_results=params.get("num_results", 5),
    )


# Register tool with the registry
def _register():
    from .registry import ToolRegistry

    ToolRegistry.register(
        name="web_search",
        description="Search the web for current information",
        parameters={
            "query": "search query",
            "categories": "optional category (general, images, news)",
            "num_results": "max results to return (default 5)",
        },
        handler=_handle_search,
        formatter=format_results_for_llm,
    )


_register()
