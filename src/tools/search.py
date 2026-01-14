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
    params = {
        "q": query,
        "format": "json",
    }
    if categories:
        params["categories"] = categories

    try:
        response = requests.get(
            config.tools.searxng_endpoint,
            params=params,
            timeout=30,
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
