"""
Query Router Package

Provides LLM-based routing to determine whether queries need
full orchestration or can be answered directly.
"""

from .router import QueryRouter, RoutingResult

__all__ = ["QueryRouter", "RoutingResult"]
