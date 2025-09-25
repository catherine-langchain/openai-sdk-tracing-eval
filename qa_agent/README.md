# Multi-Agent QA Validation System

(README written by Claude)

A sophisticated question-answer validation system using OpenAI's Agents SDK that orchestrates multiple specialized agents to validate QA pairs through planning, execution, and verification phases.

## Overview

This system validates question-answer pairs using a multi-agent architecture that:

1. **Plans** validation strategies using a planning agent
2. **Verifies** the plan quality before execution
3. **Executes** validation tasks using specialized sub-agents with tools
4. **Synthesizes** results between tasks for complex validations
5. **Makes** final verdicts with confidence scores

### Key Features

- **Multi-modal support**: Handles both text and image inputs
- **Tool-equipped agents**: Agents can use web search and code interpreters
- **Iterative refinement**: Re-plans if confidence is too low
- **Comprehensive tracing**: Tracks all agent interactions and tool usage
- **Flexible configuration**: Customize models, timeouts, and confidence thresholds

## Architecture

### Agent Types

1. **ValidationPlanner**: Creates validation strategies
2. **PlanVerifier**: Reviews and approves/rejects plans
3. **Sub-Agents** (specialized for different task types):
   - **SearchAgent**: Web search capabilities
   - **CodeAnalysisAgent**: Code interpreter for analysis
   - **FactCheckerAgent**: Fact verification via web search
   - **ComputationAgent**: Mathematical computations
   - **ReasoningAgent**: Logical reasoning
   - **SynthesizerAgent**: Processes intermediate results
4. **VerdictAgent**: Makes final validation decisions

### Validation Flow

```
Question + Answer → Planning → Verification → Execution → Verdict
                        ↑                         ↓
                        └──── Re-plan if low ─────┘
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage Example

```python
import asyncio
from qa_agent.evaluators import MultiAgentQAEvaluator, MultiAgentQAEvaluatorParameters

async def validate_cat_image():
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

    # Run validation
    result = await evaluator.evaluate(
        question=question,
        provided_answer=provided_answer,
        image_url=image_url
    )

    # Display results
    print(f"Validation Passed: {result.passed}")
    print(f"Notes: {result.notes}")

    if result.metadata.get("confidence"):
        print(f"Confidence: {result.metadata['confidence']}")

    return result

# Run the example
if __name__ == "__main__":
    result = asyncio.run(validate_cat_image())
```

## Advanced Configuration

### Custom Parameters

```python
parameters = MultiAgentQAEvaluatorParameters(
    question_field="question",           # Field name for questions
    answer_field="answer",               # Field name for answers
    planner_model="gpt-4o-mini",        # Model for planning
    executor_model="gpt-4o",             # Model for execution
    agent_timeout=90,                    # Timeout per agent (seconds)
    pass_if_low_confidence=False,       # Pass if confidence is low
    low_confidence_threshold=0.5,        # Confidence threshold
    image_field_name="image"            # Field for image URLs
)
```

### Text-Only Validation

```python
# Simple text QA validation without images
result = await evaluator.evaluate(
    question="What is the capital of France?",
    provided_answer="Paris",
    image_url=None  # No image
)
```

## Output Structure

The evaluator returns an `EvaluationResult` with:

- `passed`: Boolean indicating if validation passed
- `metadata`: Dictionary containing:
  - `confidence`: Confidence score (0.0-1.0)
  - `correct`: Whether the answer is factually correct
  - `tool_calls_summary`: Summary of all tool usage
  - `plan_verification`: Plan verification details
  - `is_low_confidence`: Low confidence flag
- `notes`: Human-readable explanation of the verdict

## Templates

The system uses Jinja2 templates for agent instructions located in the `templates/` directory:

- `planner_instructions.j2` - Planning agent instructions
- `plan_verifier_instructions.j2` - Plan verification logic
- `verdict_instructions.j2` - Final verdict generation
- Task-specific templates for each sub-agent type

## Requirements

- Python 3.10+
- OpenAI API key with access to GPT-4 models
- Dependencies listed in `requirements.txt`

## Development

### Project Structure

```
qa_agent/
├── __init__.py              # Package exports
├── evaluators.py            # Main evaluator class
├── agent_manager.py         # Agent orchestration logic
├── templates/               # Jinja2 instruction templates
│   ├── planner_instructions.j2
│   ├── verdict_instructions.j2
│   └── ... (task-specific templates)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### Running Tests

```python
# Example test script
import asyncio
from qa_agent.evaluators import MultiAgentQAEvaluator

async def test_basic_qa():
    evaluator = MultiAgentQAEvaluator()

    test_cases = [
        {
            "question": "What is 2+2?",
            "answer": "4",
            "expected": True
        },
        {
            "question": "What color is the sky?",
            "answer": "Purple",
            "expected": False
        }
    ]

    for test in test_cases:
        result = await evaluator.evaluate(
            question=test["question"],
            provided_answer=test["answer"]
        )
        assert result.passed == test["expected"], f"Test failed for: {test}"
        print(f"✓ Test passed: {test['question']}")

asyncio.run(test_basic_qa())
```

## Limitations

- Requires OpenAI API access and incurs API costs
- Agent timeout defaults to 90 seconds per agent
- Image validation requires publicly accessible image URLs
- No built-in caching mechanism for repeated validations

## License

Plz dont steal this code.
