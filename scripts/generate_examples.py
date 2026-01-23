#!/usr/bin/env python3
"""
Generate DSPy training examples for ToolOrchestrator optimization.

This script creates seed examples across different tool categories.
Run with: python -m scripts.generate_examples
"""

import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExampleGenerator:
    """Generate DSPy training examples for tool orchestration."""

    def __init__(self, output_dir: str = "data/examples"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_to_jsonl(self, examples: list[dict[str, Any]], filename: str) -> None:
        """Save examples to JSONL file."""
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
        logger.info(f"Saved {len(examples)} examples to {filepath}")

    def generate_calculate_examples(self) -> list[dict[str, Any]]:
        """Generate examples for the calculate tool (25 total)."""
        examples = [
            # Basic arithmetic (6)
            {
                "question": "What is 15 * 23?",
                "answer": "15 * 23 = 345",
                "expected_keywords": ["345"],
                "tool": "calculate",
                "category": "single_tool",
            },
            {
                "question": "Calculate 1000 / 8",
                "answer": "1000 / 8 = 125",
                "expected_keywords": ["125"],
                "tool": "calculate",
                "category": "single_tool",
            },
            {
                "question": "What is 47 + 89 + 156?",
                "answer": "47 + 89 + 156 = 292",
                "expected_keywords": ["292"],
                "tool": "calculate",
                "category": "single_tool",
            },
            {
                "question": "Calculate 999 - 567",
                "answer": "999 - 567 = 432",
                "expected_keywords": ["432"],
                "tool": "calculate",
                "category": "single_tool",
            },
            {
                "question": "What is 12 * 12 * 12?",
                "answer": "12 * 12 * 12 = 1728",
                "expected_keywords": ["1728"],
                "tool": "calculate",
                "category": "single_tool",
            },
            {
                "question": "Calculate 5678 + 4322",
                "answer": "5678 + 4322 = 10000",
                "expected_keywords": ["10000"],
                "tool": "calculate",
                "category": "single_tool",
            },
            # Powers and roots (5)
            {
                "question": "What is 2^16?",
                "answer": "2^16 = 65536",
                "expected_keywords": ["65536"],
                "tool": "calculate",
                "category": "single_tool",
            },
            {
                "question": "Calculate sqrt(144)",
                "answer": "sqrt(144) = 12",
                "expected_keywords": ["12"],
                "tool": "calculate",
                "category": "single_tool",
            },
            {
                "question": "What is 3^8?",
                "answer": "3^8 = 6561",
                "expected_keywords": ["6561"],
                "tool": "calculate",
                "category": "single_tool",
            },
            {
                "question": "Calculate the cube root of 27",
                "answer": "The cube root of 27 is 3",
                "expected_keywords": ["3"],
                "tool": "calculate",
                "category": "single_tool",
            },
            {
                "question": "What is sqrt(256)?",
                "answer": "sqrt(256) = 16",
                "expected_keywords": ["16"],
                "tool": "calculate",
                "category": "single_tool",
            },
            # Trigonometry (4)
            {
                "question": "What is sin(pi/4)?",
                "answer": "sin(pi/4) ≈ 0.7071",
                "expected_keywords": ["0.707"],
                "tool": "calculate",
                "category": "single_tool",
            },
            {
                "question": "Calculate cos(60 degrees)",
                "answer": "cos(60°) = 0.5",
                "expected_keywords": ["0.5"],
                "tool": "calculate",
                "category": "single_tool",
            },
            {
                "question": "What is tan(45 degrees)?",
                "answer": "tan(45°) = 1",
                "expected_keywords": ["1"],
                "tool": "calculate",
                "category": "single_tool",
            },
            {
                "question": "Calculate sin(30 degrees)",
                "answer": "sin(30°) = 0.5",
                "expected_keywords": ["0.5"],
                "tool": "calculate",
                "category": "single_tool",
            },
            # Statistics and logarithms (5)
            {
                "question": "What is the factorial of 10?",
                "answer": "10! = 3628800",
                "expected_keywords": ["3628800"],
                "tool": "calculate",
                "category": "single_tool",
            },
            {
                "question": "Calculate log(1000)",
                "answer": "log10(1000) = 3",
                "expected_keywords": ["3"],
                "tool": "calculate",
                "category": "single_tool",
            },
            {
                "question": "What is the natural log of e^5?",
                "answer": "ln(e^5) = 5",
                "expected_keywords": ["5"],
                "tool": "calculate",
                "category": "single_tool",
            },
            {
                "question": "Calculate 7 factorial",
                "answer": "7! = 5040",
                "expected_keywords": ["5040"],
                "tool": "calculate",
                "category": "single_tool",
            },
            {
                "question": "What is log2(1024)?",
                "answer": "log2(1024) = 10",
                "expected_keywords": ["10"],
                "tool": "calculate",
                "category": "single_tool",
            },
            # Constants and combined (5)
            {
                "question": "What is e^2?",
                "answer": "e^2 ≈ 7.389",
                "expected_keywords": ["7.38", "7.39"],
                "tool": "calculate",
                "category": "single_tool",
            },
            {
                "question": "Calculate 2 * pi * 5",
                "answer": "2 * pi * 5 ≈ 31.416 (circumference of circle with radius 5)",
                "expected_keywords": ["31.4"],
                "tool": "calculate",
                "category": "single_tool",
            },
            {
                "question": "Calculate (sqrt(16) + 2^3) / 4",
                "answer": "(sqrt(16) + 2^3) / 4 = (4 + 8) / 4 = 3",
                "expected_keywords": ["3"],
                "tool": "calculate",
                "category": "single_tool",
            },
            {
                "question": "What is pi * 3^2?",
                "answer": "pi * 3^2 = pi * 9 ≈ 28.274",
                "expected_keywords": ["28.27"],
                "tool": "calculate",
                "category": "single_tool",
            },
            {
                "question": "Calculate the area of a circle with radius 10",
                "answer": "Area = pi * r^2 = pi * 100 ≈ 314.159",
                "expected_keywords": ["314"],
                "tool": "calculate",
                "category": "single_tool",
            },
        ]
        return examples

    def generate_python_examples(self) -> list[dict[str, Any]]:
        """Generate examples for the python_execute tool (25 total)."""
        examples = [
            # Algorithms (5)
            {
                "question": "Find all prime numbers less than 50",
                "answer": "Prime numbers less than 50: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47",
                "expected_keywords": [
                    "2",
                    "3",
                    "5",
                    "7",
                    "11",
                    "13",
                    "17",
                    "19",
                    "23",
                    "29",
                    "31",
                    "37",
                    "41",
                    "43",
                    "47",
                ],
                "tool": "python_execute",
                "category": "single_tool",
            },
            {
                "question": "Sort the list [64, 34, 25, 12, 22, 11, 90] using bubble sort",
                "answer": "Sorted list: [11, 12, 22, 25, 34, 64, 90]",
                "expected_keywords": ["11", "12", "22", "25", "34", "64", "90"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            {
                "question": "Implement binary search to find 7 in [1, 3, 5, 7, 9, 11, 13]",
                "answer": "7 is found at index 3",
                "expected_keywords": ["index", "3"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            {
                "question": "Find the greatest common divisor of 48 and 18",
                "answer": "GCD of 48 and 18 is 6",
                "expected_keywords": ["6", "GCD"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            {
                "question": "Calculate the least common multiple of 12 and 18",
                "answer": "LCM of 12 and 18 is 36",
                "expected_keywords": ["36", "LCM"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            # String processing (5)
            {
                "question": "Reverse the string 'hello world' and count the vowels",
                "answer": "Reversed: 'dlrow olleh', vowels count: 3",
                "expected_keywords": ["dlrow olleh", "3"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            {
                "question": "Count the frequency of each character in 'programming'",
                "answer": "Character frequencies: p:1, r:2, o:1, g:2, a:1, m:2, i:1, n:1",
                "expected_keywords": ["r:2", "g:2", "m:2"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            {
                "question": "Check if 'racecar' is a palindrome",
                "answer": "'racecar' is a palindrome",
                "expected_keywords": ["palindrome"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            {
                "question": "Find all words longer than 5 characters in 'The quick brown fox jumps over the lazy dog'",
                "answer": "Words longer than 5 characters: ['quick', 'brown', 'jumps']",
                "expected_keywords": ["quick", "brown", "jumps"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            {
                "question": "Convert 'hello world' to title case and count words",
                "answer": "Title case: 'Hello World', word count: 2",
                "expected_keywords": ["Hello World", "2"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            # Math sequences (5)
            {
                "question": "Generate the first 10 Fibonacci numbers",
                "answer": "First 10 Fibonacci numbers: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34",
                "expected_keywords": ["0", "1", "2", "3", "5", "8", "13", "21", "34"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            {
                "question": "Calculate the sum of squares from 1 to 10",
                "answer": "Sum of squares from 1 to 10: 1+4+9+16+25+36+49+64+81+100 = 385",
                "expected_keywords": ["385"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            {
                "question": "Generate the first 8 triangular numbers",
                "answer": "First 8 triangular numbers: 1, 3, 6, 10, 15, 21, 28, 36",
                "expected_keywords": ["1", "3", "6", "10", "15", "21", "28", "36"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            {
                "question": "Calculate the sum of the first 100 natural numbers",
                "answer": "Sum of 1 to 100 = 5050",
                "expected_keywords": ["5050"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            {
                "question": "Find the 15th term of the arithmetic sequence starting at 3 with common difference 5",
                "answer": "15th term = 3 + (15-1)*5 = 3 + 70 = 73",
                "expected_keywords": ["73"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            # List operations (5)
            {
                "question": "Find the intersection of [1,2,3,4,5] and [3,4,5,6,7]",
                "answer": "Intersection: [3, 4, 5]",
                "expected_keywords": ["3", "4", "5"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            {
                "question": "Remove duplicates from [1, 2, 2, 3, 3, 3, 4, 5, 5]",
                "answer": "Unique elements: [1, 2, 3, 4, 5]",
                "expected_keywords": ["1", "2", "3", "4", "5"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            {
                "question": "Find the second largest number in [45, 23, 89, 12, 78, 56]",
                "answer": "Second largest: 78",
                "expected_keywords": ["78"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            {
                "question": "Merge and sort [5, 2, 8] and [3, 7, 1]",
                "answer": "Merged and sorted: [1, 2, 3, 5, 7, 8]",
                "expected_keywords": ["1", "2", "3", "5", "7", "8"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            {
                "question": "Count how many numbers in [1,4,6,8,10,12,15,18] are divisible by 3",
                "answer": "Numbers divisible by 3: [6, 12, 15, 18], count: 4",
                "expected_keywords": ["4"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            # Comprehensions and data structures (5)
            {
                "question": "Create a list of squares from 1 to 20",
                "answer": "Squares: [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400]",
                "expected_keywords": ["1", "4", "9", "16", "25", "36", "400"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            {
                "question": "Generate a dictionary mapping numbers 1-5 to their cubes",
                "answer": "Dictionary: {1: 1, 2: 8, 3: 27, 4: 64, 5: 125}",
                "expected_keywords": ["1", "8", "27", "64", "125"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            {
                "question": "Find all even numbers between 1 and 30",
                "answer": "Even numbers: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]",
                "expected_keywords": ["2", "4", "6", "8", "10", "30"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            {
                "question": "Create a list of tuples pairing [1,2,3] with ['a','b','c']",
                "answer": "Paired: [(1, 'a'), (2, 'b'), (3, 'c')]",
                "expected_keywords": ["(1, 'a')", "(2, 'b')", "(3, 'c')"],
                "tool": "python_execute",
                "category": "single_tool",
            },
            {
                "question": "Flatten the nested list [[1,2], [3,4], [5,6]]",
                "answer": "Flattened: [1, 2, 3, 4, 5, 6]",
                "expected_keywords": ["1", "2", "3", "4", "5", "6"],
                "tool": "python_execute",
                "category": "single_tool",
            },
        ]
        return examples

    def generate_search_examples(self) -> list[dict[str, Any]]:
        """Generate examples for the web_search tool (25 total)."""
        examples = [
            # Current events (5)
            {
                "question": "What are the latest developments in AI regulation?",
                "answer": "Recent AI regulation developments include the EU AI Act and various national frameworks addressing AI safety and ethics.",
                "expected_keywords": ["AI", "regulation", "EU"],
                "tool": "web_search",
                "category": "single_tool",
            },
            {
                "question": "What are the recent breakthroughs in fusion energy?",
                "answer": "Recent fusion energy breakthroughs include net energy gain achievements and advances in tokamak and stellarator designs.",
                "expected_keywords": ["fusion", "energy"],
                "tool": "web_search",
                "category": "single_tool",
            },
            {
                "question": "What are the latest space exploration missions?",
                "answer": "Current space missions include Mars rovers, lunar exploration programs, and commercial space ventures.",
                "expected_keywords": ["space", "mission"],
                "tool": "web_search",
                "category": "single_tool",
            },
            {
                "question": "What's happening with cryptocurrency regulations worldwide?",
                "answer": "Cryptocurrency regulations vary globally, with some countries embracing crypto while others impose strict controls.",
                "expected_keywords": ["cryptocurrency", "regulation"],
                "tool": "web_search",
                "category": "single_tool",
            },
            {
                "question": "What are the latest electric vehicle technology trends?",
                "answer": "EV trends include solid-state batteries, faster charging, and increased range capabilities.",
                "expected_keywords": ["electric", "vehicle", "battery"],
                "tool": "web_search",
                "category": "single_tool",
            },
            # Technology info (5)
            {
                "question": "What is the current stable version of Python?",
                "answer": "The current stable version of Python with its release date and key features.",
                "expected_keywords": ["Python", "version"],
                "tool": "web_search",
                "category": "single_tool",
            },
            {
                "question": "What are the new features in TypeScript 5?",
                "answer": "TypeScript 5 introduced const type parameters, decorators improvements, and performance enhancements.",
                "expected_keywords": ["TypeScript", "5"],
                "tool": "web_search",
                "category": "single_tool",
            },
            {
                "question": "What's new in React 19?",
                "answer": "React 19 includes new features like Server Components, improved suspense, and performance optimizations.",
                "expected_keywords": ["React", "19"],
                "tool": "web_search",
                "category": "single_tool",
            },
            {
                "question": "What are the latest features in Docker?",
                "answer": "Docker's latest features include improved build performance, better security, and enhanced Compose functionality.",
                "expected_keywords": ["Docker", "feature"],
                "tool": "web_search",
                "category": "single_tool",
            },
            {
                "question": "What is the latest LTS version of Node.js?",
                "answer": "The current Node.js LTS version with its support timeline and key improvements.",
                "expected_keywords": ["Node", "LTS", "version"],
                "tool": "web_search",
                "category": "single_tool",
            },
            # Facts and history (5)
            {
                "question": "Who won the Nobel Prize in Physics recently?",
                "answer": "The recent Nobel Prize in Physics was awarded for contributions to the field.",
                "expected_keywords": ["Nobel", "Physics"],
                "tool": "web_search",
                "category": "single_tool",
            },
            {
                "question": "What are the tallest buildings in the world currently?",
                "answer": "The tallest buildings include Burj Khalifa in Dubai and other supertall structures.",
                "expected_keywords": ["Burj", "tall", "building"],
                "tool": "web_search",
                "category": "single_tool",
            },
            {
                "question": "What is the current world population?",
                "answer": "The current world population is approximately 8 billion people.",
                "expected_keywords": ["billion", "population"],
                "tool": "web_search",
                "category": "single_tool",
            },
            {
                "question": "What are the largest tech companies by market cap?",
                "answer": "The largest tech companies by market cap include Apple, Microsoft, and others.",
                "expected_keywords": ["Apple", "Microsoft", "market"],
                "tool": "web_search",
                "category": "single_tool",
            },
            {
                "question": "What is the current exchange rate of USD to EUR?",
                "answer": "The current USD to EUR exchange rate fluctuates based on market conditions.",
                "expected_keywords": ["USD", "EUR", "rate"],
                "tool": "web_search",
                "category": "single_tool",
            },
            # Comparisons (5)
            {
                "question": "What are the differences between React and Vue.js?",
                "answer": "React and Vue differ in their approach to reactivity, templates, and ecosystem.",
                "expected_keywords": ["React", "Vue"],
                "tool": "web_search",
                "category": "single_tool",
            },
            {
                "question": "Compare PostgreSQL and MySQL for web applications",
                "answer": "PostgreSQL offers more advanced features while MySQL is known for simplicity and speed.",
                "expected_keywords": ["PostgreSQL", "MySQL"],
                "tool": "web_search",
                "category": "single_tool",
            },
            {
                "question": "What are the pros and cons of Kubernetes vs Docker Swarm?",
                "answer": "Kubernetes offers more features and scalability while Docker Swarm is simpler to set up.",
                "expected_keywords": ["Kubernetes", "Docker", "Swarm"],
                "tool": "web_search",
                "category": "single_tool",
            },
            {
                "question": "Compare REST and GraphQL APIs",
                "answer": "REST uses multiple endpoints while GraphQL provides a single endpoint with flexible queries.",
                "expected_keywords": ["REST", "GraphQL"],
                "tool": "web_search",
                "category": "single_tool",
            },
            {
                "question": "What's the difference between SQL and NoSQL databases?",
                "answer": "SQL databases are relational with fixed schemas while NoSQL databases offer flexible schemas.",
                "expected_keywords": ["SQL", "NoSQL"],
                "tool": "web_search",
                "category": "single_tool",
            },
            # How-to (5)
            {
                "question": "How do you deploy a Docker container to Kubernetes?",
                "answer": "Deploy to Kubernetes using kubectl apply with a deployment manifest YAML file.",
                "expected_keywords": ["kubectl", "deploy", "Kubernetes"],
                "tool": "web_search",
                "category": "single_tool",
            },
            {
                "question": "How do you set up CI/CD with GitHub Actions?",
                "answer": "Set up GitHub Actions by creating workflow YAML files in .github/workflows directory.",
                "expected_keywords": ["GitHub", "Actions", "workflow"],
                "tool": "web_search",
                "category": "single_tool",
            },
            {
                "question": "How do you implement OAuth 2.0 authentication?",
                "answer": "OAuth 2.0 involves registering an app, redirecting users, and exchanging tokens.",
                "expected_keywords": ["OAuth", "token", "authentication"],
                "tool": "web_search",
                "category": "single_tool",
            },
            {
                "question": "How do you optimize a PostgreSQL database?",
                "answer": "Optimize PostgreSQL with indexing, query analysis, configuration tuning, and vacuuming.",
                "expected_keywords": ["PostgreSQL", "index", "optimize"],
                "tool": "web_search",
                "category": "single_tool",
            },
            {
                "question": "How do you implement WebSockets in a Node.js application?",
                "answer": "Implement WebSockets using the ws library or Socket.io for real-time communication.",
                "expected_keywords": ["WebSocket", "Node", "Socket"],
                "tool": "web_search",
                "category": "single_tool",
            },
        ]
        return examples

    def generate_delegate_reasoner_examples(self) -> list[dict[str, Any]]:
        """Generate examples for ask_reasoner delegate (15 total)."""
        examples = [
            # Complex analysis (5)
            {
                "question": "Analyze the pros and cons of microservices vs monolithic architecture",
                "answer": "Microservices offer scalability and independent deployment but add complexity. Monoliths are simpler but harder to scale.",
                "expected_keywords": ["microservices", "monolith", "scalability"],
                "tool": "ask_reasoner",
                "category": "delegate",
            },
            {
                "question": "What are the trade-offs between eventual consistency and strong consistency?",
                "answer": "Strong consistency ensures immediate data accuracy but reduces availability. Eventual consistency improves performance but allows temporary inconsistencies.",
                "expected_keywords": ["consistency", "availability", "trade-off"],
                "tool": "ask_reasoner",
                "category": "delegate",
            },
            {
                "question": "Explain the CAP theorem and its implications for distributed systems",
                "answer": "CAP theorem states distributed systems can only guarantee two of: Consistency, Availability, Partition tolerance.",
                "expected_keywords": [
                    "CAP",
                    "consistency",
                    "availability",
                    "partition",
                ],
                "tool": "ask_reasoner",
                "category": "delegate",
            },
            {
                "question": "What factors should be considered when choosing between SQL and NoSQL databases?",
                "answer": "Consider data structure, scalability needs, ACID requirements, query complexity, and team expertise.",
                "expected_keywords": ["SQL", "NoSQL", "ACID", "scalability"],
                "tool": "ask_reasoner",
                "category": "delegate",
            },
            {
                "question": "Analyze the impact of technical debt on software development",
                "answer": "Technical debt slows development, increases bugs, and reduces maintainability. Regular refactoring is essential.",
                "expected_keywords": ["technical", "debt", "refactoring"],
                "tool": "ask_reasoner",
                "category": "delegate",
            },
            # Multi-step reasoning (5)
            {
                "question": "If all programmers are logical, and Alice is a programmer, what can we conclude about Alice?",
                "answer": "Alice is logical, based on the logical deduction from the given premises.",
                "expected_keywords": ["Alice", "logical"],
                "tool": "ask_reasoner",
                "category": "delegate",
            },
            {
                "question": "A company has 100 employees. 60% work remotely, 40% are engineers. If 30% of engineers work remotely, what percentage of remote workers are engineers?",
                "answer": "12 engineers work remotely out of 60 remote workers, so 20% of remote workers are engineers.",
                "expected_keywords": ["20%", "engineers", "remote"],
                "tool": "ask_reasoner",
                "category": "delegate",
            },
            {
                "question": "Explain step by step how to debug a memory leak in a Node.js application",
                "answer": "Use heap snapshots, memory profilers, identify retention paths, and fix references preventing garbage collection.",
                "expected_keywords": ["heap", "memory", "profiler", "garbage"],
                "tool": "ask_reasoner",
                "category": "delegate",
            },
            {
                "question": "Walk me through the process of designing a rate limiting system",
                "answer": "Consider token bucket or sliding window algorithms, storage (Redis), configuration, and handling of exceeded limits.",
                "expected_keywords": ["token", "bucket", "Redis", "limit"],
                "tool": "ask_reasoner",
                "category": "delegate",
            },
            {
                "question": "How would you approach migrating a monolithic database to microservices?",
                "answer": "Start with bounded contexts, use strangler fig pattern, implement data synchronization, and migrate incrementally.",
                "expected_keywords": ["strangler", "bounded", "context", "incremental"],
                "tool": "ask_reasoner",
                "category": "delegate",
            },
            # Strategy (5)
            {
                "question": "What's the best approach to refactor a 10k line legacy codebase?",
                "answer": "Add tests first, identify seams, extract modules incrementally, maintain backwards compatibility.",
                "expected_keywords": ["test", "refactor", "incremental"],
                "tool": "ask_reasoner",
                "category": "delegate",
            },
            {
                "question": "How should a startup prioritize feature development with limited resources?",
                "answer": "Use impact vs effort matrix, focus on MVP, gather user feedback, iterate quickly.",
                "expected_keywords": ["MVP", "impact", "effort", "feedback"],
                "tool": "ask_reasoner",
                "category": "delegate",
            },
            {
                "question": "What strategy would you recommend for handling high availability requirements?",
                "answer": "Implement redundancy, load balancing, failover mechanisms, health checks, and disaster recovery.",
                "expected_keywords": ["redundancy", "failover", "load balancing"],
                "tool": "ask_reasoner",
                "category": "delegate",
            },
            {
                "question": "How do you design an effective code review process?",
                "answer": "Define guidelines, use automated checks, limit PR size, focus on constructive feedback, and track metrics.",
                "expected_keywords": ["review", "PR", "feedback", "automated"],
                "tool": "ask_reasoner",
                "category": "delegate",
            },
            {
                "question": "What approach would you take to improve application performance?",
                "answer": "Profile first, optimize bottlenecks, use caching, optimize queries, and consider horizontal scaling.",
                "expected_keywords": ["profile", "bottleneck", "cache", "optimize"],
                "tool": "ask_reasoner",
                "category": "delegate",
            },
        ]
        return examples

    def generate_delegate_coder_examples(self) -> list[dict[str, Any]]:
        """Generate examples for ask_coder delegate (10 total)."""
        examples = [
            # Code review (4)
            {
                "question": "Review this function for bugs: def add(a, b): return a - b",
                "answer": "Bug found: The function is named 'add' but performs subtraction. Should use '+' instead of '-'.",
                "expected_keywords": ["bug", "subtraction", "+"],
                "tool": "ask_coder",
                "category": "delegate",
            },
            {
                "question": "Is this code thread-safe? counter = 0; def increment(): global counter; counter += 1",
                "answer": "Not thread-safe. The increment operation is not atomic. Use locks or atomic operations.",
                "expected_keywords": ["thread", "safe", "atomic", "lock"],
                "tool": "ask_coder",
                "category": "delegate",
            },
            {
                "question": "Review this SQL for injection vulnerabilities: query = f'SELECT * FROM users WHERE id = {user_id}'",
                "answer": "SQL injection vulnerability. Use parameterized queries instead of string formatting.",
                "expected_keywords": ["SQL", "injection", "parameterized"],
                "tool": "ask_coder",
                "category": "delegate",
            },
            {
                "question": "Check this Python code for issues: with open('file.txt') as f: data = f.read(); return data",
                "answer": "Syntax error: 'return' cannot be used outside a function. Also consider error handling for file operations.",
                "expected_keywords": ["return", "function", "error"],
                "tool": "ask_coder",
                "category": "delegate",
            },
            # Debugging (3)
            {
                "question": "Why might this code cause a memory leak: while True: data.append(fetch_data())",
                "answer": "The list grows indefinitely without clearing. Data is never removed, causing memory exhaustion.",
                "expected_keywords": ["memory", "leak", "grows", "clear"],
                "tool": "ask_coder",
                "category": "delegate",
            },
            {
                "question": "Debug: for i in range(len(lst)): if lst[i] == target: lst.remove(lst[i])",
                "answer": "Modifying list while iterating causes index issues. Use list comprehension or iterate backwards.",
                "expected_keywords": ["modify", "iterate", "index"],
                "tool": "ask_coder",
                "category": "delegate",
            },
            {
                "question": "Why does this return None: def greet(name): print(f'Hello {name}')",
                "answer": "The function uses 'print' instead of 'return'. Print outputs to console but returns None.",
                "expected_keywords": ["print", "return", "None"],
                "tool": "ask_coder",
                "category": "delegate",
            },
            # Generation (3)
            {
                "question": "Write a Python decorator that logs function execution time",
                "answer": "A decorator using functools.wraps that records start time, calls function, calculates duration, and logs it.",
                "expected_keywords": ["decorator", "time", "functools", "log"],
                "tool": "ask_coder",
                "category": "delegate",
            },
            {
                "question": "Create a Python context manager for database transactions",
                "answer": "A context manager that begins transaction on __enter__, commits on success, and rolls back on exception.",
                "expected_keywords": ["context", "manager", "commit", "rollback"],
                "tool": "ask_coder",
                "category": "delegate",
            },
            {
                "question": "Write a retry decorator with exponential backoff",
                "answer": "A decorator that catches exceptions, waits with exponential delay, and retries up to a max count.",
                "expected_keywords": ["retry", "exponential", "backoff", "exception"],
                "tool": "ask_coder",
                "category": "delegate",
            },
        ]
        return examples

    def generate_delegate_fast_examples(self) -> list[dict[str, Any]]:
        """Generate examples for ask_fast delegate (5 total)."""
        examples = [
            {
                "question": "Summarize in 5 words: The quick brown fox jumps over the lazy dog",
                "answer": "Fox jumps over lazy dog.",
                "expected_keywords": ["fox", "jumps", "dog"],
                "tool": "ask_fast",
                "category": "delegate",
            },
            {
                "question": "Convert to uppercase: hello world",
                "answer": "HELLO WORLD",
                "expected_keywords": ["HELLO", "WORLD"],
                "tool": "ask_fast",
                "category": "delegate",
            },
            {
                "question": "Extract the numbers from: 'I have 3 apples and 5 oranges'",
                "answer": "3, 5",
                "expected_keywords": ["3", "5"],
                "tool": "ask_fast",
                "category": "delegate",
            },
            {
                "question": "Translate 'hello' to French",
                "answer": "Bonjour",
                "expected_keywords": ["Bonjour"],
                "tool": "ask_fast",
                "category": "delegate",
            },
            {
                "question": "What's the sentiment of: 'This product is amazing!'",
                "answer": "Positive sentiment",
                "expected_keywords": ["positive"],
                "tool": "ask_fast",
                "category": "delegate",
            },
        ]
        return examples

    def generate_multi_tool_examples(self) -> list[dict[str, Any]]:
        """Generate examples requiring multiple tools (20 total)."""
        examples = [
            # Search + Calculate (5)
            {
                "question": "Find the current US population and calculate 1% of it",
                "answer": "US population is approximately 330 million, 1% is about 3.3 million.",
                "expected_keywords": ["330", "million", "3.3"],
                "tools": ["web_search", "calculate"],
                "category": "multi_tool",
            },
            {
                "question": "What is the distance to the Moon in km, and how long would it take light to travel there?",
                "answer": "Moon is ~384,400 km away. Light takes about 1.28 seconds to reach it.",
                "expected_keywords": ["384", "second", "light"],
                "tools": ["web_search", "calculate"],
                "category": "multi_tool",
            },
            {
                "question": "Find the height of Mount Everest and calculate what percentage of it is above 8000m",
                "answer": "Everest is 8,849m. Above 8000m is 849m, which is about 9.6% of total height.",
                "expected_keywords": ["8849", "849", "9.6"],
                "tools": ["web_search", "calculate"],
                "category": "multi_tool",
            },
            {
                "question": "What's the speed of sound in m/s, and how long does it take to travel 1 km?",
                "answer": "Speed of sound is ~343 m/s. It takes about 2.9 seconds to travel 1 km.",
                "expected_keywords": ["343", "2.9", "seconds"],
                "tools": ["web_search", "calculate"],
                "category": "multi_tool",
            },
            {
                "question": "Find Earth's circumference and calculate how many kilometers in 1 degree of longitude at the equator",
                "answer": "Earth's circumference is ~40,075 km. 1 degree = 40075/360 ≈ 111.3 km.",
                "expected_keywords": ["40075", "111", "degree"],
                "tools": ["web_search", "calculate"],
                "category": "multi_tool",
            },
            # Search + Python (4)
            {
                "question": "Find Python's list sorting algorithm and implement a simple version",
                "answer": "Python uses Timsort. Here's a simplified merge sort implementation.",
                "expected_keywords": ["Timsort", "merge", "sort"],
                "tools": ["web_search", "python_execute"],
                "category": "multi_tool",
            },
            {
                "question": "What's the formula for calculating BMI, then calculate it for 70kg and 175cm",
                "answer": "BMI = weight(kg) / height(m)^2. For 70kg and 1.75m: 70/1.75^2 ≈ 22.9",
                "expected_keywords": ["BMI", "22.9"],
                "tools": ["web_search", "python_execute"],
                "category": "multi_tool",
            },
            {
                "question": "Find the Fibonacci sequence formula and generate the first 15 numbers",
                "answer": "Fibonacci: F(n) = F(n-1) + F(n-2). First 15: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377",
                "expected_keywords": ["Fibonacci", "377"],
                "tools": ["web_search", "python_execute"],
                "category": "multi_tool",
            },
            {
                "question": "Search for the quadratic formula and solve x^2 - 5x + 6 = 0",
                "answer": "Using quadratic formula: x = (5 ± sqrt(25-24))/2. Solutions: x = 3 and x = 2.",
                "expected_keywords": ["quadratic", "3", "2"],
                "tools": ["web_search", "python_execute"],
                "category": "multi_tool",
            },
            # Calculate + Reasoner (3)
            {
                "question": "Calculate compound interest on $10000 at 7% for 10 years, then explain the power of compounding",
                "answer": "A = 10000 * (1.07)^10 ≈ $19,672. Compounding allows interest to earn interest, accelerating growth.",
                "expected_keywords": ["19672", "compound", "interest"],
                "tools": ["calculate", "ask_reasoner"],
                "category": "multi_tool",
            },
            {
                "question": "Calculate 2^64 and explain why this number is significant in computing",
                "answer": "2^64 = 18,446,744,073,709,551,616. This is the max value for 64-bit unsigned integers.",
                "expected_keywords": ["18446744073709551616", "64-bit"],
                "tools": ["calculate", "ask_reasoner"],
                "category": "multi_tool",
            },
            {
                "question": "Calculate the factorial of 20 and explain where factorials are used in programming",
                "answer": "20! = 2,432,902,008,176,640,000. Factorials are used in permutations, combinations, and probability.",
                "expected_keywords": [
                    "2432902008176640000",
                    "permutation",
                    "combination",
                ],
                "tools": ["calculate", "ask_reasoner"],
                "category": "multi_tool",
            },
            # Python + Coder (3)
            {
                "question": "Write a function to check if a number is prime, then have it reviewed for efficiency",
                "answer": "Basic prime check can be optimized by only checking up to sqrt(n) and skipping even numbers.",
                "expected_keywords": ["prime", "sqrt", "efficient"],
                "tools": ["python_execute", "ask_coder"],
                "category": "multi_tool",
            },
            {
                "question": "Implement a simple stack data structure and get feedback on the implementation",
                "answer": "Stack implementation with push, pop, peek methods. Review suggests adding size limits and error handling.",
                "expected_keywords": ["stack", "push", "pop"],
                "tools": ["python_execute", "ask_coder"],
                "category": "multi_tool",
            },
            {
                "question": "Write code to parse JSON and handle errors, then review for best practices",
                "answer": "Use json.loads with try/except for JSONDecodeError. Review suggests validation and logging.",
                "expected_keywords": ["json", "parse", "exception"],
                "tools": ["python_execute", "ask_coder"],
                "category": "multi_tool",
            },
            # Search + Reasoner (3)
            {
                "question": "Find information about quantum computing and explain it in simple terms",
                "answer": "Quantum computers use qubits that can be in superposition. Like having multiple calculations at once.",
                "expected_keywords": ["quantum", "qubit", "superposition"],
                "tools": ["web_search", "ask_reasoner"],
                "category": "multi_tool",
            },
            {
                "question": "Search for blockchain technology and analyze its use cases beyond cryptocurrency",
                "answer": "Blockchain enables supply chain tracking, voting systems, digital identity, and smart contracts.",
                "expected_keywords": ["blockchain", "supply chain", "smart contract"],
                "tools": ["web_search", "ask_reasoner"],
                "category": "multi_tool",
            },
            {
                "question": "Find recent AI developments and analyze their potential impact on software development",
                "answer": "AI assistants can generate code, but developers are still needed for architecture, review, and complex logic.",
                "expected_keywords": ["AI", "code", "developer"],
                "tools": ["web_search", "ask_reasoner"],
                "category": "multi_tool",
            },
            # Triple tool combinations (2)
            {
                "question": "Search for mortgage interest rates, calculate monthly payment for $300k at that rate over 30 years, then explain how amortization works",
                "answer": "At ~7% rate, monthly payment is ~$1,996. Early payments are mostly interest; principal portion grows over time.",
                "expected_keywords": ["mortgage", "monthly", "amortization"],
                "tools": ["web_search", "calculate", "ask_reasoner"],
                "category": "multi_tool",
            },
            {
                "question": "Find the algorithm for calculating day of week from date, implement it in Python, and have the code reviewed",
                "answer": "Zeller's congruence or similar algorithm. Implementation uses modular arithmetic. Review checks edge cases.",
                "expected_keywords": ["Zeller", "day", "week"],
                "tools": ["web_search", "python_execute", "ask_coder"],
                "category": "multi_tool",
            },
        ]
        return examples

    def generate_routing_examples(self) -> list[dict[str, Any]]:
        """Generate routing examples that don't need tools (25 total)."""
        examples = [
            # Greetings (5)
            {
                "query": "Hello",
                "needs_tools": False,
                "reasoning": "Simple greeting that can be answered directly",
                "direct_answer": "Hello! How can I help you today?",
                "category": "greeting",
            },
            {
                "query": "Good morning",
                "needs_tools": False,
                "reasoning": "Simple greeting",
                "direct_answer": "Good morning! How can I assist you today?",
                "category": "greeting",
            },
            {
                "query": "Hi there!",
                "needs_tools": False,
                "reasoning": "Casual greeting",
                "direct_answer": "Hi! What can I do for you?",
                "category": "greeting",
            },
            {
                "query": "Thanks!",
                "needs_tools": False,
                "reasoning": "Simple acknowledgment",
                "direct_answer": "You're welcome!",
                "category": "greeting",
            },
            {
                "query": "Goodbye",
                "needs_tools": False,
                "reasoning": "Farewell message",
                "direct_answer": "Goodbye! Feel free to come back if you have more questions.",
                "category": "greeting",
            },
            # Basic knowledge (5)
            {
                "query": "What color is the sky?",
                "needs_tools": False,
                "reasoning": "Basic factual knowledge",
                "direct_answer": "The sky appears blue due to the scattering of sunlight by the atmosphere.",
                "category": "basic_knowledge",
            },
            {
                "query": "How many days are in a week?",
                "needs_tools": False,
                "reasoning": "Common knowledge",
                "direct_answer": "There are 7 days in a week.",
                "category": "basic_knowledge",
            },
            {
                "query": "What is the capital of Japan?",
                "needs_tools": False,
                "reasoning": "Basic geography knowledge",
                "direct_answer": "The capital of Japan is Tokyo.",
                "category": "basic_knowledge",
            },
            {
                "query": "How many planets are in our solar system?",
                "needs_tools": False,
                "reasoning": "Basic astronomy knowledge",
                "direct_answer": "There are 8 planets in our solar system.",
                "category": "basic_knowledge",
            },
            {
                "query": "What is the boiling point of water?",
                "needs_tools": False,
                "reasoning": "Basic science knowledge",
                "direct_answer": "Water boils at 100°C (212°F) at standard atmospheric pressure.",
                "category": "basic_knowledge",
            },
            # Definitions (5)
            {
                "query": "What is a variable in programming?",
                "needs_tools": False,
                "reasoning": "Basic programming concept",
                "direct_answer": "A variable is a named storage location in a program that holds a value which can be modified during execution.",
                "category": "definition",
            },
            {
                "query": "What is recursion?",
                "needs_tools": False,
                "reasoning": "Programming concept explanation",
                "direct_answer": "Recursion is when a function calls itself to solve a problem by breaking it into smaller subproblems.",
                "category": "definition",
            },
            {
                "query": "What is an API?",
                "needs_tools": False,
                "reasoning": "Technical term definition",
                "direct_answer": "An API (Application Programming Interface) is a set of protocols that allow different software applications to communicate.",
                "category": "definition",
            },
            {
                "query": "What is a database?",
                "needs_tools": False,
                "reasoning": "Technical concept",
                "direct_answer": "A database is an organized collection of structured data stored electronically and accessed via a management system.",
                "category": "definition",
            },
            {
                "query": "What does HTTP stand for?",
                "needs_tools": False,
                "reasoning": "Acronym explanation",
                "direct_answer": "HTTP stands for Hypertext Transfer Protocol, used for transmitting web pages over the internet.",
                "category": "definition",
            },
            # Simple math (5)
            {
                "query": "What is 2 + 2?",
                "needs_tools": False,
                "reasoning": "Simple arithmetic from knowledge",
                "direct_answer": "2 + 2 equals 4.",
                "category": "simple_math",
            },
            {
                "query": "What is 10 times 10?",
                "needs_tools": False,
                "reasoning": "Simple multiplication",
                "direct_answer": "10 times 10 equals 100.",
                "category": "simple_math",
            },
            {
                "query": "What is half of 20?",
                "needs_tools": False,
                "reasoning": "Simple division",
                "direct_answer": "Half of 20 is 10.",
                "category": "simple_math",
            },
            {
                "query": "What is 100 minus 1?",
                "needs_tools": False,
                "reasoning": "Simple subtraction",
                "direct_answer": "100 minus 1 equals 99.",
                "category": "simple_math",
            },
            {
                "query": "What is a dozen?",
                "needs_tools": False,
                "reasoning": "Common number term",
                "direct_answer": "A dozen is 12 items.",
                "category": "simple_math",
            },
            # Opinions/advice (5)
            {
                "query": "Should I learn Python or JavaScript first?",
                "needs_tools": False,
                "reasoning": "General advice question",
                "direct_answer": "Both are great choices. Python is often recommended for beginners due to its readable syntax, while JavaScript is essential for web development.",
                "category": "advice",
            },
            {
                "query": "Is it better to use tabs or spaces for indentation?",
                "needs_tools": False,
                "reasoning": "Opinion-based coding preference",
                "direct_answer": "This is a matter of preference. Spaces offer consistent rendering, while tabs allow customizable width. Most important is consistency within a project.",
                "category": "advice",
            },
            {
                "query": "What's a good way to learn programming?",
                "needs_tools": False,
                "reasoning": "General learning advice",
                "direct_answer": "Start with a beginner-friendly language, work through tutorials, build small projects, and practice consistently.",
                "category": "advice",
            },
            {
                "query": "How can I improve my code quality?",
                "needs_tools": False,
                "reasoning": "General programming advice",
                "direct_answer": "Write tests, review others' code, follow style guides, refactor regularly, and keep functions small and focused.",
                "category": "advice",
            },
            {
                "query": "What IDE should I use?",
                "needs_tools": False,
                "reasoning": "Tool recommendation",
                "direct_answer": "Popular choices include VS Code (versatile, free), PyCharm (great for Python), and JetBrains IDEs. Try a few to find what suits you.",
                "category": "advice",
            },
        ]
        return examples

    def generate_all(self) -> dict[str, int]:
        """Generate all examples and save to files."""
        counts = {}

        # Generate orchestration examples
        calculate = self.generate_calculate_examples()
        self.save_to_jsonl(calculate, "calculate_examples.jsonl")
        counts["calculate"] = len(calculate)

        python_examples = self.generate_python_examples()
        self.save_to_jsonl(python_examples, "python_examples.jsonl")
        counts["python"] = len(python_examples)

        search = self.generate_search_examples()
        self.save_to_jsonl(search, "search_examples.jsonl")
        counts["search"] = len(search)

        reasoner = self.generate_delegate_reasoner_examples()
        self.save_to_jsonl(reasoner, "delegate_reasoner.jsonl")
        counts["delegate_reasoner"] = len(reasoner)

        coder = self.generate_delegate_coder_examples()
        self.save_to_jsonl(coder, "delegate_coder.jsonl")
        counts["delegate_coder"] = len(coder)

        fast = self.generate_delegate_fast_examples()
        self.save_to_jsonl(fast, "delegate_fast.jsonl")
        counts["delegate_fast"] = len(fast)

        multi = self.generate_multi_tool_examples()
        self.save_to_jsonl(multi, "multi_tool_examples.jsonl")
        counts["multi_tool"] = len(multi)

        # Generate routing examples
        routing = self.generate_routing_examples()
        self.save_to_jsonl(routing, "routing_no_tools.jsonl")
        counts["routing"] = len(routing)

        # Summary
        total = sum(counts.values())
        logger.info(f"Total examples generated: {total}")
        for category, count in counts.items():
            logger.info(f"  {category}: {count}")

        return counts


def main():
    """Main entry point."""
    generator = ExampleGenerator()
    counts = generator.generate_all()
    print(
        f"\nGenerated {sum(counts.values())} examples across {len(counts)} categories"
    )


if __name__ == "__main__":
    main()
