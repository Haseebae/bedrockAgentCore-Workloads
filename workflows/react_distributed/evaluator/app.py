"""
Evaluator Sub-Agent — Evaluates plan execution results.
Uses LangGraph with a single evaluator node and structured output.
Deployed as an independent Bedrock AgentCore runtime.
"""
import json
import os
import sys
import logging
import contextvars
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END, MessagesState
from bedrock_agentcore.runtime import BedrockAgentCoreApp

from common.logging_callback import SessionMetricsCallback

from prompts import EVALUATOR_PROMPT

# ==================== LOGGING ====================

metric_logger = logging.getLogger("agent_metrics")
metric_logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(message)s'))
metric_logger.addHandler(handler)

session_id_var = contextvars.ContextVar("session_id", default="unknown_session")
current_node_var = contextvars.ContextVar("current_node", default="evaluator")
trace_id_var = contextvars.ContextVar("trace_id", default="unknown_trace")
state_id_var = contextvars.ContextVar("state_id", default="unknown_state")

# ==================== STRUCTURED OUTPUT ====================

class EvalResult(BaseModel):
    success: bool = Field(..., description="Did the action fully address the user query?")
    needs_retry: bool = Field(..., description="Should we try another iteration with a new plan?")
    reason: str = Field(..., description="Brief explanation of the evaluation.")
    feedback: Optional[str] = Field(
        default=None,
        description="If needs_retry=true, specific feedback on why plan failed and what should change.",
    )

# ==================== AGENT STATE & GRAPH ====================

class EvaluatorState(MessagesState):
    original_task: str
    plan: str
    execution: str
    iteration_count: Optional[int]
    max_iterations: Optional[int]
    step_count: Optional[int]
    evaluation: Optional[dict]


def build_agent():
    model_name = os.environ.get("MODEL_NAME", "openai:gpt-4o-mini")
    model = init_chat_model(model_name)
    evaluator_model = model.with_structured_output(EvalResult)

    def evaluator_node(state: EvaluatorState):
        current_node_var.set("evaluator")
        messages = state["messages"]
        step_count = (state.get("step_count") or 0) + 1
        plan_json = state.get("plan", "{}")
        iteration_count = state.get("iteration_count", 1)
        max_iterations = state.get("max_iterations", 3)

        result_json = state.get("execution", "")

        system_msg_content = EVALUATOR_PROMPT.format(
            plan_json=plan_json,
            result_json=result_json,
            iteration_count=iteration_count,
            max_iterations=max_iterations
        )
        system_msg = SystemMessage(content=system_msg_content)
        eval_result = evaluator_model.invoke([system_msg] + messages)

        metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "debug",
            "node_name": "evaluator",
            "iteration_count": iteration_count,
            "step_count": step_count,
            "trace_id": trace_id_var.get(),
            "session_id": session_id_var.get(),
            "state_id": state_id_var.get(),
            "request": system_msg_content,
            "response": str(eval_result.model_dump())[:1000]
        }))

        return {
            "evaluation": eval_result.model_dump(),
            "step_count": step_count,
        }

    graph = StateGraph(EvaluatorState)
    graph.add_node("evaluator", evaluator_node)
    graph.add_edge(START, "evaluator")
    graph.add_edge("evaluator", END)

    return graph.compile()


# ==================== ENTRYPOINT ====================

_agent = None

def _get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


app = BedrockAgentCoreApp()

@app.entrypoint
def handle(payload):
    """
    Payload contract:
      IN:  {original_task, plan, execution, session_id, trace_id, state_id, actor_id,
            memory_config, agent_state: {iteration_count, step_count}}
      OUT: {response: JSON {status, feedback, success, needs_retry, reason},
            agent_state: {iteration_count, step_count}}
    """
    original_task = payload.get("original_task", "")
    plan = payload.get("plan", "")
    execution = payload.get("execution", "")
    session_id = payload.get("session_id", "default_session_id")
    trace_id = payload.get("trace_id", uuid.uuid4().hex)
    state_id = payload.get("state_id", uuid.uuid4().hex)
    memory_config = payload.get("memory_config", "empty")
    agent_state = payload.get("agent_state", {"iteration_count": 0, "step_count": 0})

    session_id_var.set(session_id)
    trace_id_var.set(trace_id)
    state_id_var.set(state_id)
    current_node_var.set("evaluator")

    if not os.environ.get("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY not set"}

    agent = _get_agent()

    config = {
        "callbacks": [SessionMetricsCallback(
            session_id=session_id,
            current_node_var=current_node_var,
            metric_logger=metric_logger,
            trace_id_var=trace_id_var,
            state_id_var=state_id_var,
        )],
        "configurable": {"session_id": session_id}
    }

    iteration_count = agent_state.get("iteration_count", 1)
    max_iterations = 3

    result = agent.invoke({
        "messages": [HumanMessage(content=original_task)],
        "original_task": original_task,
        "plan": plan,
        "execution": execution,
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "step_count": agent_state.get("step_count", 0),
    }, config=config)

    evaluation = result.get("evaluation", {})

    # Map to status format expected by orchestrator
    status = "approved" if evaluation.get("success") and not evaluation.get("needs_retry") else "needs_revision"

    updated_agent_state = {
        "iteration_count": iteration_count,
        "step_count": result.get("step_count", agent_state.get("step_count", 0)),
    }

    return {
        "response": json.dumps({
            "status": status,
            "feedback": evaluation.get("feedback", ""),
            "success": evaluation.get("success", False),
            "needs_retry": evaluation.get("needs_retry", False),
            "reason": evaluation.get("reason", ""),
        }),
        "agent_state": updated_agent_state,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/Users/haseeb/Code/iisc/bedrockAC/workflows/react_distributed/evaluator/.env.dev")
    app.run()