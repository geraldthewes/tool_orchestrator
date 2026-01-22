"""
DSPy Signature for Query Routing.

Determines if a query requires tool orchestration or can be answered directly.
"""

import dspy


class QueryRouting(dspy.Signature):
    """Analyze a query to determine if it requires tools or can be answered directly.

    You are a query router. Analyze the user's query and determine which tool is most
    relevant to answer the query.

    Rules:
    1. If the query benefits from any tool (current info, calculations, code execution,
       web search, external LLM analysis), set needs_tools to True.
    2. Only set needs_tools to False if the answer is simple, short, and obvious from
       your knowledge.

    Examples requiring tools (needs_tools=True):
    - "What's the weather today?" - Needs current info (web search)
    - "Calculate 2^100" - Needs calculation tool
    - "Search for latest news on AI" - Needs web search
    - "Run this Python code" - Needs code execution
    - "Analyze this text" - Needs external LLM
    - "Create a poem about cats" - Needs external LLM for quality output

    Examples NOT requiring tools (needs_tools=False):
    - "Hello" - Simple greeting
    - "What is the capital of France?" - Basic factual knowledge
    - "Thanks!" - Acknowledgment
    """

    query: str = dspy.InputField(desc="The user's query to analyze")
    available_tools: str = dspy.InputField(
        desc="List of available tools and their descriptions"
    )

    needs_tools: bool = dspy.OutputField(
        desc="True if the query requires tools, False if it can be answered directly"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of why tools are or aren't needed"
    )
    direct_answer: str = dspy.OutputField(
        desc="If needs_tools is False, provide the direct answer here. Otherwise empty string."
    )
