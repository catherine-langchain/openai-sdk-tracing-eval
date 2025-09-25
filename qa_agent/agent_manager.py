import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import agents
from agents import (
    Agent,
    AgentOutputSchemaBase,
    CodeInterpreterTool,
    ModelSettings,
    OpenAIResponsesModel,
    RunConfig,
    Runner,
    Tool,
    WebSearchTool,
)
from agents.tracing.scope import Scope
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field
from openai import AsyncOpenAI


# Structured outputs for planning
class TaskType(str, Enum):
    SEARCH = "search"
    CODE_ANALYSIS = "code_analysis"
    FACT_CHECK = "fact_check"
    COMPUTATION = "computation"
    REASONING = "reasoning"
    SYNTHESIS = "synthesis"


class ValidationTask(BaseModel):
    task_id: str = Field(description="Unique identifier for this task")
    task_type: TaskType = Field(description="Type of validation task")
    description: str = Field(description="What this task should accomplish")
    priority: int = Field(description="Priority level (1=highest, 5=lowest)")
    depends_on: list[str] = Field(default=[], description="Task IDs this depends on")
    agent_instructions: str = Field(description="Specific instructions for the agent")


class ValidationPlan(BaseModel):
    strategy: str = Field(description="Overall validation strategy")
    tasks: list[ValidationTask] = Field(description="list of validation tasks")
    confidence_threshold: float = Field(description="Minimum confidence required")
    max_iterations: int = Field(default=2, description="Maximum planning iterations")


class SubAgentResult(BaseModel):
    task_id: str
    success: bool
    findings: str
    confidence: float
    evidence: list[str] = Field(default=[])


class InputTemplatedAgent:
    def __init__(
        self,
        agent_name: str,
        model: str,
        instructions: str,
        output_type: type[Any] | AgentOutputSchemaBase,
        template_name: str,
        tools: list[Tool] | None = None,
        template_dir: Path = Path(__file__).parent / "templates",
    ) -> None:
        # Create ModelSettings with temperature=0 for deterministic outputs
        model_settings = ModelSettings(temperature=0)
        if model == "gpt-4o-mini":
            # o4-mini models may have different temperature settings
            model_settings = ModelSettings()

        # Do not use mutable data structures for argument defaults
        if tools is None:
            tools = []

        # Create OpenAI client - no portkey wrapper
        openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.agent = Agent(
            name=agent_name,
            model=OpenAIResponsesModel(model=model, openai_client=openai_client),
            instructions=instructions,
            output_type=output_type,
            tools=tools,
            model_settings=model_settings,
        )
        self._template = self._load_template(template_name, template_dir)

    def _load_template(self, template_name: str, template_dir: Path) -> Any:
        """Load Jinja2 template from file."""

        if not (template_dir / f"{template_name}.j2").exists():
            raise FileNotFoundError(
                f"Template {template_dir}/{template_name}.j2 not found"
            )

        env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)
        template = env.get_template(f"{template_name}.j2")
        return template

    def render_template(self, **kwargs: Any) -> str:
        """Render Jinja2 template with provided context."""
        return self._template.render(**kwargs)


class FinalVerdict(BaseModel):
    correct: bool = Field(description="Whether the provided answer is correct")
    confidence: float = Field(description="Confidence score (0.0-1.0)")
    rationale: str = Field(description="Detailed reasoning for the verdict")
    actual_answer: str | None = Field(
        default=None, description="Correct answer if original was wrong"
    )
    actual_answer_explanation: str | None = Field(
        default=None, description="Explanation of correct answer"
    )


class PlanVerificationResult(BaseModel):
    approved: bool = Field(description="Whether the plan is approved for execution")
    confidence: float = Field(description="Confidence in the plan quality (0.0-1.0)")
    feedback: str = Field(description="Detailed feedback on the plan")
    suggestions: list[str] = Field(
        default=[], description="Specific suggestions for improvement"
    )
    critical_issues: list[str] = Field(
        default=[], description="Critical issues that must be addressed"
    )


class QAValidationToolCallSummary(BaseModel):
    total_tool_calls: int
    code_interpreter_calls: int
    web_search_calls: int
    all_tool_calls: list[dict]
    code_interpreter_details: list[dict]
    web_search_details: list[dict]


class QAValidationResult(BaseModel):
    correct: bool
    confidence: float | None = None
    rationale: str | None = None
    actual_answer: str | None = None
    actual_answer_explanation: str | None = None
    plan_verification: PlanVerificationResult | None = None
    tool_calls_summary: QAValidationToolCallSummary | None = None


@dataclass
class ValidationContext:
    question: str
    question_id: str
    provided_answer: str
    explanation: str | None = None
    reference: str | None = None
    validation_plan: ValidationPlan | None = None
    plan_verification: PlanVerificationResult | None = None
    agent_results: list[SubAgentResult] = field(default_factory=list)
    iteration_count: int = 0
    plan_iteration_count: int = 0
    image_url: str | None = None

    def get_max_iterations(self) -> int:
        """Get max iterations with default fallback."""
        if self.validation_plan is not None:
            return self.validation_plan.max_iterations
        return 2  # Default max iterations

    def get_confidence_threshold(self) -> float:
        """Get confidence threshold with default fallback."""
        if self.validation_plan is not None:
            return self.validation_plan.confidence_threshold
        return 0.5  # Default confidence threshold

    def get_tasks(self) -> list[ValidationTask]:
        """Get tasks safely with empty list fallback."""
        if self.validation_plan is not None and hasattr(self.validation_plan, "tasks"):
            return self.validation_plan.tasks
        return []  # Return empty list if no validation plan or no tasks

    def to_data_for_rendering(self) -> dict[str, str | None]:
        """Convert context to data for rendering."""
        data = {
            "question_id": self.question_id,
            "question": self.question,
            "provided_answer": self.provided_answer,
            "explanation": self.explanation,
            "reference": self.reference,
        }

        # Add image URL if it exists
        if self.image_url:
            data["image_url"] = self.image_url

        return data


class QAValidationAgentManager:
    """Main agent manager orchestrating the validation process."""

    MAX_PLAN_ITERATIONS = 3  # Maximum attempts to create an approved plan
    SOURCE_NAME = "QAValidationAgentManager"

    def __init__(
        self,
        planner_model: str = "gpt-4o-mini",
        executor_model: str = "gpt-4o",
        agent_timeout: int = 90,
    ) -> None:
        """
        Initialize the QAValidationAgentManager.

        Args:
            planner_model: The model to use for planning.
            executor_model: The model to use for execution.
            agent_timeout: The timeout for each agent call (in seconds). Defaults to 90 seconds.
        """
        self.planner_model = planner_model
        self.executor_model = executor_model
        self._agent_timeout = agent_timeout
        self.sub_agents: dict[TaskType, InputTemplatedAgent] = {}
        self._initialize_agents()

    def _format_agent_input(
        self, text: str, image_url: str | None = None
    ) -> list[dict[str, Any]]:
        """Format input for OpenAI Agents SDK with proper message structure.

        Args:
            text: The text content for the message
            image_url: Optional image URL for multimodal input

        Returns:
            List containing a properly formatted message for the OpenAI Agents SDK
        """
        content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]

        if image_url:
            content.append({"type": "input_image", "image_url": image_url})

        return [{"type": "message", "role": "user", "content": content}]

    def _log_trace(
        self,
        traces: list[dict],
        agent_name: str,
        input_data: list,
        output_data: Any,
        duration: float,
        tool_calls: list[dict] | None = None,
    ) -> None:
        """Log agent call trace to the provided traces list."""
        trace = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "input": str(input_data),  # Convert to string for logging
            "output": str(output_data),
            "duration_seconds": duration,
        }
        if tool_calls:
            trace["tool_calls"] = tool_calls
        traces.append(trace)

    def _parse_tool_calls(self, result: Any) -> list[dict]:
        tool_calls = []
        if hasattr(result, "new_items"):
            for item in result.new_items:
                if hasattr(item, "type") and item.type == "tool_call_item":
                    tool_call = {
                        "type": "tool_call",
                        "call_id": getattr(item.raw_item, "id", "unknown"),
                    }

                    # Add tool-specific details in simple format
                    if hasattr(item.raw_item, "code"):
                        tool_call.update(
                            {
                                "tool_type": "code_interpreter",
                                "code": str(item.raw_item.code),
                                "container_id": str(
                                    getattr(item.raw_item, "container_id", "")
                                ),
                                "status": str(getattr(item.raw_item, "status", "")),
                            }
                        )
                    elif hasattr(item.raw_item, "query"):
                        tool_call.update(
                            {
                                "tool_type": "web_search",
                                "query": str(item.raw_item.query),
                            }
                        )
                    else:
                        # For other tool types, capture all available attributes as strings
                        for attr_name in dir(item.raw_item):
                            if not attr_name.startswith("_") and not callable(
                                getattr(item.raw_item, attr_name)
                            ):
                                try:
                                    attr_value = getattr(item.raw_item, attr_name)
                                    # Convert everything to string for serialization
                                    tool_call[attr_name] = str(attr_value)
                                except Exception:
                                    tool_call[attr_name] = "Error getting attribute"

                    tool_calls.append(tool_call)
        return tool_calls

    def _get_run_config(self) -> RunConfig | None:
        """Get run config for the current trace."""
        # Simplified - no portkey integration
        return None

    async def _execute_agent_with_timeout(
        self,
        traces: list[dict],
        agent: Agent,
        agent_input: list,
    ) -> Any:
        """Execute an agent with timeout and comprehensive error handling."""
        start_time = time.time()
        agent_name = agent.name

        run_config = self._get_run_config()

        try:
            # Use regular run to get complete result
            result = await asyncio.wait_for(
                Runner.run(agent, agent_input, run_config=run_config),
                timeout=float(self._agent_timeout),
            )

            duration = time.time() - start_time

            # Extract all tool call data in simple serializable format
            tool_calls = self._parse_tool_calls(result)

            self._log_trace(
                traces,
                agent_name,
                agent_input,
                result.final_output,
                duration,
                tool_calls,
            )
            return result.final_output

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            error_msg = f"TIMEOUT: {agent_name} timed out after {self._agent_timeout}s"
            self._log_trace(traces, agent_name, agent_input, error_msg, duration, [])
            raise asyncio.TimeoutError(error_msg) from None

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"ERROR: {agent_name} failed: {str(e)}"
            self._log_trace(traces, agent_name, agent_input, error_msg, duration, [])
            raise Exception(error_msg) from e

    def _initialize_agents(self) -> None:
        """Initialize all agents in the validation system with appropriate tools."""

        code_interpreter_config = {
            "type": "code_interpreter",
            "container": {"type": "auto"},
        }

        # 1. Planner Agent - Creates validation strategy
        self.planner_agent = InputTemplatedAgent(
            agent_name="ValidationPlanner",
            model=self.planner_model,
            instructions="You are a ValidationPlanner. Wait for context to be provided.",
            output_type=ValidationPlan,
            template_name="planner_instructions",
        )

        # 2. Plan Verifier Agent - Reviews and approves/rejects plans
        self.plan_verifier_agent = InputTemplatedAgent(
            agent_name="PlanVerifier",
            model=self.executor_model,
            instructions="You are a PlanVerifier. Wait for context to be provided.",
            output_type=PlanVerificationResult,
            template_name="plan_verifier_instructions",
        )

        # 3. Sub-Agents for specialized tasks with tools
        self.sub_agents[TaskType.SEARCH] = InputTemplatedAgent(
            agent_name="SearchAgent",
            model=self.executor_model,
            instructions="You are a SearchAgent. Wait for context to be provided.",
            output_type=SubAgentResult,
            tools=[WebSearchTool()],
            template_name="search_agent_instructions",
        )

        self.sub_agents[TaskType.CODE_ANALYSIS] = InputTemplatedAgent(
            agent_name="CodeAnalysisAgent",
            model=self.executor_model,
            instructions="You are a CodeAnalysisAgent. Wait for context to be provided.",
            output_type=SubAgentResult,
            tools=[CodeInterpreterTool(code_interpreter_config)],
            template_name="code_analysis_instructions",
        )

        self.sub_agents[TaskType.FACT_CHECK] = InputTemplatedAgent(
            agent_name="FactCheckerAgent",
            model=self.executor_model,
            instructions="You are a FactCheckerAgent. Wait for context to be provided.",
            output_type=SubAgentResult,
            tools=[WebSearchTool()],
            template_name="fact_checker_instructions",
        )

        self.sub_agents[TaskType.COMPUTATION] = InputTemplatedAgent(
            agent_name="ComputationAgent",
            model=self.executor_model,
            instructions="You are a ComputationAgent. Wait for context to be provided.",
            output_type=SubAgentResult,
            tools=[CodeInterpreterTool(code_interpreter_config)],
            template_name="computation_instructions",
        )

        self.sub_agents[TaskType.REASONING] = InputTemplatedAgent(
            agent_name="ReasoningAgent",
            model=self.executor_model,
            instructions="You are a ReasoningAgent. Wait for context to be provided.",
            output_type=SubAgentResult,
            template_name="reasoning_instructions",
        )

        # 4. Synthesizer Agent - Processes and synthesizes results between tasks
        self.sub_agents[TaskType.SYNTHESIS] = InputTemplatedAgent(
            agent_name="SynthesizerAgent",
            model=self.executor_model,
            instructions="You are a SynthesizerAgent. Wait for context to be provided.",
            output_type=SubAgentResult,
            template_name="synthesis_instructions",
        )

        # 5. Final Verdict Agent - Makes final decision
        self.verdict_agent = InputTemplatedAgent(
            agent_name="VerdictAgent",
            model=self.executor_model,
            instructions="You are a VerdictAgent. Wait for context to be provided.",
            output_type=FinalVerdict,
            template_name="verdict_instructions",
        )

    async def _update_context_with_verified_plan(
        self, traces: list[dict], context: ValidationContext
    ) -> None:
        """
        Update the context with a new validation plan.
        """
        previous_feedback: None | str = None

        while context.plan_iteration_count < self.MAX_PLAN_ITERATIONS:
            # Create or recreate validation plan
            context.validation_plan = await self._create_validation_plan(
                traces, context, previous_results=context.agent_results
            )

            # Verify the plan
            context.plan_verification = await self._verify_validation_plan(
                traces, context, previous_feedback=previous_feedback
            )

            # If plan is approved, proceed to execution
            if (
                context.plan_verification is not None
                and context.plan_verification.approved
            ):
                return

            # If plan is not approved, prepare feedback for next iteration
            if context.plan_verification is not None:
                previous_feedback = context.plan_verification.feedback

            context.plan_iteration_count += 1

        msg = (
            f"Failed to create an approved verified plan after {self.MAX_PLAN_ITERATIONS} iterations. "
            "Continue with a low-confidence plan."
        )
        print(f"Warning: {msg}")

    async def validate_qa_pair(
        self,
        question: str,
        provided_answer: str,
        question_id: str = "unknown",
        explanation: str | None = None,
        reference: str | None = None,
        image_url: str | None = None,
    ) -> QAValidationResult:
        """
        Main validation method using the agent loop.

        Args:
            question: The question to be validated
            provided_answer: The answer provided by the user.
            question_id: The unique identifier for the question.
            explanation: The explanation for the answer.
            reference: The reference for the answer.
            image_url: Optional image URL for multimodal evaluation.

        Returns:
            Dictionary with validation results
        """
        with agents.trace("validate_qa_pair"):
            return await self._validate_qa_pair_with_trace(
                question,
                provided_answer,
                question_id,
                explanation,
                reference,
                image_url,
            )

    async def _validate_qa_pair_with_trace(
        self,
        question: str,
        provided_answer: str,
        question_id: str = "unknown",
        explanation: str | None = None,
        reference: str | None = None,
        image_url: str | None = None,
    ) -> QAValidationResult:
        # Create local traces for this validation
        traces: list[dict] = []

        context = ValidationContext(
            question_id=question_id,
            question=question,
            provided_answer=provided_answer,
            explanation=explanation,
            reference=reference,
            agent_results=[],
            image_url=image_url,
        )

        try:
            # Phase 1: Planning with Verification Loop
            await self._update_context_with_verified_plan(traces, context)

            # Phase 2: Execution Loop
            while context.iteration_count < context.get_max_iterations():
                context.agent_results = await self._execute_validation_plan(
                    traces, context
                )

                # Phase 3: Verdict
                verdict = await self._make_final_verdict(traces, context)

                # Check if we're confident enough
                if verdict.confidence >= context.get_confidence_threshold():
                    return self._format_result(traces, verdict, context)

                # If not confident enough, replan (with verification)
                context.iteration_count += 1
                if context.iteration_count < context.get_max_iterations():
                    # Reset plan iteration counter for replanning
                    context.plan_iteration_count = 0

                    # Replanning with verification loop
                    await self._update_context_with_verified_plan(traces, context)

            # If we've exhausted iterations, return the current result
            verdict = await self._make_final_verdict(traces, context)
            return self._format_result(traces, verdict, context)

        except Exception as e:
            # Return error result with current traces
            print(f"Error: Validation failed due to error: {str(e)}. Returning")
            verdict = FinalVerdict(
                correct=False,
                confidence=0.0,
                rationale=f"Validation failed due to error: {str(e)}",
            )
            # Include traces even for errors
            return self._format_result(traces, verdict, context)

    async def _create_validation_plan(
        self,
        traces: list[dict],
        context: ValidationContext,
        previous_results: list[SubAgentResult] | None = None,
    ) -> ValidationPlan:
        """Create a validation plan using the planner agent."""

        # Render the planner instructions
        planning_input_text = self.planner_agent.render_template(
            data=context.to_data_for_rendering(),
            previous_results=previous_results if previous_results is not None else [],
            iteration_count=context.iteration_count,
        )

        # Format input with proper message structure
        planning_input = self._format_agent_input(
            planning_input_text, context.image_url
        )

        # Get validation plan with error handling
        try:
            return await self._execute_agent_with_timeout(
                traces,
                self.planner_agent.agent,
                planning_input,
            )
        except (asyncio.TimeoutError, Exception) as e:
            # Create a fallback plan if planner fails
            print(
                f"Error: Failed to create validation plan. Using fallback verification plan with reasoning: {str(e)}"
            )
            return ValidationPlan(
                strategy="Fallback validation plan due to planner failure",
                tasks=[
                    ValidationTask(
                        task_id="fallback_reasoning",
                        task_type=TaskType.REASONING,
                        description="Basic reasoning validation",
                        priority=1,
                        depends_on=[],
                        agent_instructions="Perform basic logical validation of the answer",
                    )
                ],
                confidence_threshold=0.5,
                max_iterations=1,
            )

    async def _verify_validation_plan(
        self,
        traces: list[dict],
        context: ValidationContext,
        previous_feedback: str | None = None,
    ) -> PlanVerificationResult:
        """Verify the validation plan using the plan verifier agent."""

        # Render the plan verifier instructions
        verification_input_text = self.plan_verifier_agent.render_template(
            data=context.to_data_for_rendering(),
            validation_plan=context.validation_plan,
            previous_feedback=previous_feedback,
            iteration_count=context.plan_iteration_count,
        )

        # Format input with proper message structure
        verification_input = self._format_agent_input(
            verification_input_text, context.image_url
        )

        # Get plan verification with error handling
        try:
            return await self._execute_agent_with_timeout(
                traces,
                self.plan_verifier_agent.agent,
                verification_input,
            )
        except (asyncio.TimeoutError, Exception) as e:
            # Create a fallback approval if verifier fails
            print(
                f"Error: Failed to verify validation plan. Proceeding with default approval: {str(e)}"
            )
            return PlanVerificationResult(
                approved=True,
                confidence=0.5,
                feedback=f"Plan verification failed: {str(e)}. Proceeding with default approval.",
                suggestions=[],
                critical_issues=[],
            )

    async def _execute_validation_plan(
        self, traces: list[dict], context: ValidationContext
    ) -> list[SubAgentResult]:
        """Execute the validation plan using appropriate agents with synthesis steps between tasks."""

        # Sort tasks by dependencies
        tasks = context.get_tasks()
        sorted_tasks: list[ValidationTask] = (
            self._sort_tasks_by_dependencies(tasks) if tasks else []
        )

        results: list[SubAgentResult] = []
        completed_tasks = set()

        for i, task in enumerate(sorted_tasks):
            # Check if dependencies are satisfied
            if all(dep in completed_tasks for dep in task.depends_on):
                # Execute task
                try:
                    input_templated_agent = self.sub_agents[task.task_type]
                    agent_input = self._prepare_agent_input(task, context, results)

                    task_result = await self._execute_agent_with_timeout(
                        traces,
                        input_templated_agent.agent,
                        agent_input,
                    )

                    # Store result
                    result = SubAgentResult(
                        task_id=task.task_id,
                        success=True,
                        findings=str(task_result),
                        confidence=0.8,  # Default confidence
                        evidence=[],
                    )
                    results.append(result)
                    completed_tasks.add(task.task_id)

                    # Add synthesis step after each task (except the last one)
                    # Only add synthesis for complex validations with many tasks to avoid overhead
                    if (
                        i < len(sorted_tasks) - 1 and len(sorted_tasks) >= 4
                    ):  # Not the last task and has 4+ tasks
                        synthesis_result = await self._execute_synthesis_step(
                            traces,
                            context,
                            results,
                            sorted_tasks[i + 1 :],
                            task.task_id,
                        )
                        if synthesis_result:
                            results.append(synthesis_result)

                except (asyncio.TimeoutError, Exception) as e:
                    # Error already logged by _execute_agent_with_timeout
                    print(
                        f"Error: Task failed. task_id:{task.task_id}, task_type:{task.task_type}, error:{str(e)}"
                    )

                    # Handle failure gracefully
                    results.append(
                        SubAgentResult(
                            task_id=task.task_id,
                            success=False,
                            findings=f"Task failed: {str(e)}",
                            confidence=0.0,
                            evidence=[],
                        )
                    )
                    completed_tasks.add(task.task_id)

        return results

    async def _execute_synthesis_step(
        self,
        traces: list[dict],
        context: ValidationContext,
        completed_results: list[SubAgentResult],
        upcoming_tasks: list[ValidationTask],
        last_task_id: str,
    ) -> SubAgentResult | None:
        """Execute a synthesis step to process results and provide context for upcoming tasks."""

        try:
            # Prepare synthesis input
            synthesis_input_text = self.sub_agents[TaskType.SYNTHESIS].render_template(
                data=context.to_data_for_rendering(),
                validation_plan=context.validation_plan,
                completed_results=completed_results,
                upcoming_tasks=upcoming_tasks,
            )

            # Format input with proper message structure
            synthesis_input = self._format_agent_input(
                synthesis_input_text, context.image_url
            )

            # Execute synthesis with timeout to prevent hanging
            synthesis_result = await self._execute_agent_with_timeout(
                traces,
                self.sub_agents[TaskType.SYNTHESIS].agent,
                synthesis_input,
            )

            # Create synthesis result
            synthesis_task_id = f"synthesis_after_{last_task_id}"
            return SubAgentResult(
                task_id=synthesis_task_id,
                success=True,
                findings=str(synthesis_result),
                confidence=getattr(synthesis_result, "confidence", 0.7),
                evidence=getattr(synthesis_result, "evidence", []),
            )

        except (asyncio.TimeoutError, Exception):
            # Log synthesis failure but don't stop the process
            # Error already logged by _execute_agent_with_timeout
            return None

    async def _make_final_verdict(
        self, traces: list[dict], context: ValidationContext
    ) -> FinalVerdict:
        """Make final verdict using the verdict agent."""

        # Render the verdict instructions
        verdict_input_text = self.verdict_agent.render_template(
            data=context.to_data_for_rendering(),
            validation_plan=context.validation_plan,
            agent_results=context.agent_results,
            confidence_threshold=context.get_confidence_threshold(),
        )

        # Format input with proper message structure
        verdict_input = self._format_agent_input(verdict_input_text, context.image_url)

        # Get final verdict
        try:
            return await self._execute_agent_with_timeout(
                traces,
                self.verdict_agent.agent,
                verdict_input,
            )
        except (asyncio.TimeoutError, Exception) as e:
            print(f"Error: Failed to make final verdict: {str(e)}")
            # If verdict fails, create a default low-confidence verdict
            return FinalVerdict(
                correct=False,
                confidence=0.1,
                rationale=f"Verdict generation failed: {str(e)}",
                actual_answer=None,
                actual_answer_explanation=None,
            )

    def _sort_tasks_by_dependencies(
        self, tasks: list[ValidationTask]
    ) -> list[ValidationTask]:
        """Sort tasks respecting dependencies and priority."""
        # Simple topological sort + priority
        sorted_tasks: list[ValidationTask] = []
        remaining_tasks = tasks.copy()

        while remaining_tasks:
            # Find tasks with no pending dependencies
            ready_tasks = [
                task
                for task in remaining_tasks
                if all(
                    dep_id in [t.task_id for t in sorted_tasks]
                    for dep_id in task.depends_on
                )
            ]

            if not ready_tasks:
                # Break circular dependencies by taking highest priority
                ready_tasks = [min(remaining_tasks, key=lambda t: t.priority)]

            # Sort ready tasks by priority
            ready_tasks.sort(key=lambda t: t.priority)

            # Add the highest priority ready task
            next_task = ready_tasks[0]
            sorted_tasks.append(next_task)
            remaining_tasks.remove(next_task)

        return sorted_tasks

    def _prepare_agent_input(
        self,
        task: ValidationTask,
        context: ValidationContext,
        previous_results: list[SubAgentResult],
    ) -> list:
        """Prepare input for a specific agent task using Jinja2 templates."""

        # Filter relevant previous results
        relevant_results = [
            result for result in previous_results if result.task_id in task.depends_on
        ]

        # TODO: This comes from the original implementation. Confirm whether this is expected or not
        if task.task_type == TaskType.SYNTHESIS:
            agent = self.sub_agents[TaskType.REASONING]
        else:
            agent = self.sub_agents[task.task_type]

        text_input = agent.render_template(
            data=context.to_data_for_rendering(),
            task_id=task.task_id,
            task_description=task.description,
            task_priority=task.priority,
            agent_instructions=task.agent_instructions,
            previous_results=relevant_results,
        )

        # Format input with proper message structure
        return self._format_agent_input(text_input, context.image_url)

    def _format_result(
        self, traces: list[dict], verdict: FinalVerdict, context: ValidationContext
    ) -> QAValidationResult:
        """Format the final result for the app."""

        all_tool_calls, code_interpreter_calls, web_search_calls = [], [], []

        for trace in traces:
            agent_name = trace.get("agent", "Unknown")
            tool_calls = trace.get("tool_calls", [])

            for tool_call in tool_calls:
                tool_call_info = {
                    "agent": agent_name,
                    "timestamp": trace.get("timestamp", ""),
                    "duration": trace.get("duration_seconds", 0),
                    **tool_call,  # Include all tool call data
                }

                all_tool_calls.append(tool_call_info)

                # Categorize by tool type
                if tool_call.get("tool_type") == "code_interpreter":
                    code_interpreter_calls.append(tool_call_info)
                elif tool_call.get("tool_type") == "web_search":
                    web_search_calls.append(tool_call_info)

        tool_calls_summary = QAValidationToolCallSummary(
            total_tool_calls=len(all_tool_calls),
            code_interpreter_calls=len(code_interpreter_calls),
            web_search_calls=len(web_search_calls),
            all_tool_calls=all_tool_calls,
            code_interpreter_details=code_interpreter_calls,
            web_search_details=web_search_calls,
        )

        result = QAValidationResult(
            correct=verdict.correct,
            confidence=verdict.confidence,
            rationale=verdict.rationale,
            actual_answer=verdict.actual_answer,
            actual_answer_explanation=verdict.actual_answer_explanation,
            plan_verification=context.plan_verification,
            tool_calls_summary=tool_calls_summary,
        )

        return result