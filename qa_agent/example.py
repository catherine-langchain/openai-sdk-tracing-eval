#!/usr/bin/env python3
"""
Example script demonstrating the Multi-Agent QA Validation System.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path to import qa_agent
sys.path.insert(0, str(Path(__file__).parent.parent))

from qa_agent.evaluators import MultiAgentQAEvaluator, MultiAgentQAEvaluatorParameters


async def validate_cat_image():
    """Example: Validate a cat image with question-answer pair."""
    print("=" * 60)
    print("Multi-Agent QA Validation Example")
    print("=" * 60)
    print()

    # Configure the evaluator
    parameters = MultiAgentQAEvaluatorParameters(
        planner_model="gpt-4o-mini",
        executor_model="gpt-4o",
        agent_timeout=90,
        low_confidence_threshold=0.5
    )

    # Create evaluator instance
    evaluator = MultiAgentQAEvaluator(parameters)

    # Example: Cat image validation
    question = "What type of cat is shown in this image?"
    provided_answer = "This is a tuxedo cat with black and white fur pattern."
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/George%2C_a_perfect_example_of_a_tuxedo_cat.jpg/1250px-George%2C_a_perfect_example_of_a_tuxedo_cat.jpg"

    print(f"Question: {question}")
    print(f"Provided Answer: {provided_answer}")
    print(f"Image URL: {image_url[:50]}...")
    print()
    print("Running validation...")
    print("-" * 40)

    # Run validation
    result = await evaluator.evaluate(
        question=question,
        provided_answer=provided_answer,
        image_url=image_url
    )

    # Display results
    print()
    print("Results:")
    print("-" * 40)
    print(f"✓ Validation Passed: {result.passed}")
    print(f"✓ Notes: {result.notes}")

    if result.metadata.get("confidence"):
        print(f"✓ Confidence: {result.metadata['confidence']}")

    if result.metadata.get("correct"):
        print(f"✓ Answer Correct: {result.metadata['correct']}")

    print()
    print("=" * 60)

    return result


async def validate_simple_qa():
    """Example: Simple text-only QA validation."""
    print()
    print("Simple Text QA Validation Example")
    print("-" * 40)

    evaluator = MultiAgentQAEvaluator()

    question = "What is the capital of France?"
    provided_answer = "Paris"

    print(f"Question: {question}")
    print(f"Provided Answer: {provided_answer}")
    print()
    print("Running validation...")

    result = await evaluator.evaluate(
        question=question,
        provided_answer=provided_answer,
        image_url=None
    )

    print(f"✓ Validation Passed: {result.passed}")
    print(f"✓ Confidence: {result.metadata.get('confidence', 'N/A')}")
    print()

    return result


async def main():
    """Run all examples."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    try:
        # Run cat image example
        await validate_cat_image()

        # Optionally run simple text example
        # await validate_simple_qa()

    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())