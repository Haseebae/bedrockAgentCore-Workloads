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
import time
import psutil
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END, MessagesState
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# ----- IMPORT FOR NATIVE AGENTCORE MEMORY -----
from langgraph_checkpoint_aws import AgentCoreMemorySaver

from common.logging_callback import SessionMetricsCallback

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
local_state_id_var = contextvars.ContextVar("local_state_id", default="unknown_local_state")
local_trace_id_var = contextvars.ContextVar("local_trace_id", default="unknown_local_trace")

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
    evaluation: Optional[dict[str, Any]]
    step_count: Optional[int]
    iteration_count: Optional[int]


def build_agent():
    model_name = os.environ.get("MODEL_NAME", "openai:gpt-4o-mini")
    model = init_chat_model(model_name)
    evaluator_model = model.with_structured_output(EvalResult)

    def evaluator_node(state: EvaluatorState):
        current_node_var.set("evaluator")
        messages = state["messages"]
        step_count = state["step_count"] + 1
        iteration_count = state["iteration_count"]

        eval_result = evaluator_model.invoke(messages)

        metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "debug",
            "node_name": "evaluator",
            "iteration_count": iteration_count,
            "step_count": step_count,
            "trace_id": trace_id_var.get(),
            "session_id": session_id_var.get(),
            "state_id": state_id_var.get(),
            "local_state_id": local_state_id_var.get(),
            "local_trace_id": local_trace_id_var.get(),
            "message_len": len(messages),
            "request": "",
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

    memory_id = os.environ.get("MEMORY_ID")
    aws_region = os.environ.get("AWS_REGION", "ap-south-1")

    if not memory_id:
        raise ValueError("MEMORY_ID environment variable is missing.")

    checkpointer = AgentCoreMemorySaver(memory_id, region_name=aws_region)
    return graph.compile(checkpointer=checkpointer)


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
      IN:  {original_task, plan, execution, session_id, trace_id, orchestrator_state_id,
            actor_id, memory_config, thread_id, workload_type, s3_enabled,
            agent_state: {iteration_count, step_count}}
      OUT: {response: JSON {status, feedback, success, needs_retry, reason},
            agent_state: {iteration_count, step_count}}
    """
    start_time = time.time()
    original_task = payload.get("original_task", "")
    session_id = payload.get("session_id", "default_session_id")
    trace_id = payload.get("trace_id", uuid.uuid4().hex)
    orchestrator_state_id = payload.get("orchestrator_state_id", uuid.uuid4().hex)
    local_state_id = uuid.uuid4().hex
    local_trace_id = payload.get("local_trace_id", uuid.uuid4().hex)
    memory_config = payload.get("memory_config", "empty")
    thread_id = payload.get("thread_id", trace_id)
    actor_id = payload.get("actor_id", "default_actor_id")
    agent_state = payload.get("agent_state", {"iteration_count": 0, "step_count": 0})

    session_id_var.set(session_id)
    trace_id_var.set(trace_id)
    state_id_var.set(orchestrator_state_id)
    local_state_id_var.set(local_state_id)
    local_trace_id_var.set(local_trace_id)
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
        "configurable": {
            "session_id": session_id,
            "thread_id": thread_id,
            "actor_id": actor_id,
        }
    }

    iteration_count = agent_state.get("iteration_count", 1)

    # Store initial memory using psutil
    process = psutil.Process()
    initial_mem = process.memory_info().rss

    result = agent.invoke({
        "messages": [HumanMessage(content=original_task)],
        "iteration_count": iteration_count,
        "step_count": agent_state.get("step_count", 0),
        "evaluation": None,
    }, config=config)

    evaluation = result.get("evaluation", {})

    # Map to status format expected by orchestrator
    status = "approved" if evaluation.get("success") and not evaluation.get("needs_retry") else "needs_revision"

    updated_agent_state = {
        "iteration_count": iteration_count,
        "step_count": result.get("step_count", agent_state.get("step_count", 0)),
    }

    # Capture peak memory using psutil
    final_mem = process.memory_info().rss
    peak_mem = max(initial_mem, final_mem)
    peak_memory_gb = peak_mem / (1024 * 1024 * 1024)

    end_time = time.time()
    wall_clock_time = end_time - start_time

    metric_logger.info(json.dumps({
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
        "event_type": "billing_metrics",
        "session_id": session_id,
        "trace_id": trace_id,
        "state_id": orchestrator_state_id,
        "local_state_id": local_state_id,
        "local_trace_id": local_trace_id,
        "peak_memory_gb": round(peak_memory_gb, 4),
        "step_count": result.get("step_count", 0),
        "wall_clock_s": round(wall_clock_time, 4)
    }))

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