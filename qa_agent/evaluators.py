from typing import Optional
from pydantic import BaseModel, Field
from .agent_manager import (
    QAValidationAgentManager,
    QAValidationResult,
)


class EvaluationResult(BaseModel):
    """Simple evaluation result model replacing snorkel_daas.EvaluationResult"""
    passed: bool
    metadata: dict[str, str] = Field(default_factory=dict)
    notes: str = ""


class MultiAgentQAEvaluatorParameters(BaseModel):
    """Configuration for the MultiAgentQAEvaluator.

    Attributes:
        question_field: The field name in the task data that contains the question.
        answer_field: The field name in the task data that contains the answer.
        planner_model: The model to use for planning.
        executor_model: The model to use for execution.
        agent_timeout: The timeout for each agent call (in seconds).
        pass_if_low_confidence: Whether to pass if the confidence is below the threshold.
        low_confidence_threshold: The threshold to determine if the confidence is low.
        image_field_name: Optional field containing image URL for multimodal evaluation.
    """

    question_field: str = "question"
    answer_field: str = "answer"
    planner_model: str = Field(default="gpt-4o-mini")
    executor_model: str = Field(default="gpt-4o")
    agent_timeout: int = Field(default=90, ge=1, le=3600)
    pass_if_low_confidence: bool = Field(default=False)
    low_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    image_field_name: Optional[str] = None


class MultiAgentQAEvaluator:
    """Multi-agent Evaluator for validating question-answer pairs using the OpenAI Agents SDK.

    This evaluator uses a multi-agent system to validate question-answer pairs.
    The system consists of a question-answering agent and a validation agent.
    The question-answering agent is responsible for answering the question.
    The validation agent is responsible for validating the answer.
    The evaluator uses the OpenAI Agents SDK to create the agents and the system.
    """

    def __init__(
        self,
        parameters: Optional[MultiAgentQAEvaluatorParameters] = None,
    ) -> None:
        if parameters is None:
            parameters = MultiAgentQAEvaluatorParameters()

        self._config = parameters
        self._agent_manager = QAValidationAgentManager(
            planner_model=parameters.planner_model,
            executor_model=parameters.executor_model,
            agent_timeout=parameters.agent_timeout,
        )

    async def evaluate(
        self,
        question: str,
        provided_answer: str,
        image_url: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate a question-answer pair.

        Args:
            question: The question to evaluate
            provided_answer: The answer to validate
            image_url: Optional image URL for multimodal evaluation

        Returns:
            EvaluationResult with passed status and metadata
        """
        result = await self._agent_manager.validate_qa_pair(
            question=question,
            provided_answer=provided_answer,
            image_url=image_url,
        )

        # Build metadata from results
        metadata: dict[str, str] = {}
        if result.tool_calls_summary:
            metadata["tool_calls_summary"] = result.tool_calls_summary.model_dump_json()
        if result.plan_verification:
            metadata["plan_verification"] = result.plan_verification.model_dump_json()
        if result.confidence:
            metadata["confidence"] = str(result.confidence)
        if result.correct:
            metadata["correct"] = str(result.correct)

        # Check if the confidence is low
        is_low_confidence = (
            result.confidence is not None
            and result.confidence < self._config.low_confidence_threshold
        )
        metadata["is_low_confidence"] = str(is_low_confidence)

        # Check the if_low_confidence flag and the confidence is less than the threshold, we pass the evaluation.
        passed = result.correct
        if self._config.pass_if_low_confidence and is_low_confidence:
            print(
                f"Passed the evaluation because the confidence is lower than the threshold. "
                f"({result.confidence} < {self._config.low_confidence_threshold})"
            )
            passed = True

        notes = self._get_notes(result, is_low_confidence, passed)

        evaluation_result = EvaluationResult(
            passed=passed,
            metadata=metadata,
            notes=notes,
        )

        return evaluation_result

    def _get_notes(
        self,
        result: QAValidationResult,
        is_low_confidence: bool,
        passed: bool,
    ) -> str:
        verdict = "correct" if result.correct else "incorrect"
        details = f"The provided answer is {verdict} with confidence {result.confidence}. The rationale follows: {result.rationale}."

        if self._config.pass_if_low_confidence and is_low_confidence:
            details += " However, the evaluator passed the evaluation because the confidence is low."

        if not result.correct:
            details += f" The correct answer is {result.actual_answer}. The actual answer rationale follows: {result.actual_answer_explanation}"

        return details