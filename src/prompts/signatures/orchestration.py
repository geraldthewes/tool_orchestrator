"""
DSPy Signature for Tool Orchestration.

Defines the input/output specification for ReAct-style tool orchestration.
"""

import dspy


class ToolOrchestrationTask(dspy.Signature):
    """Orchestrate tools to answer a question using ReAct reasoning.

    You are an AI assistant with access to tools. Use them to gather information
    and compute results needed to answer the question.

    Follow the ReAct pattern:
    1. Think about what you need to do
    2. Select and use an appropriate tool
    3. Observe the result
    4. Repeat until you have enough information
    5. Provide a final answer

    Important rules:
    - Use tools when you need external information or computation
    - Delegate complex tasks to appropriate expert LLMs
    - Be concise but thorough in your final answers
    """

    question: str = dspy.InputField(desc="The user's question or task to answer")
    answer: str = dspy.OutputField(
        desc="The final answer after using tools to gather necessary information"
    )
