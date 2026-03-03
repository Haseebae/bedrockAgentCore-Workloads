"""
ReAct Distributed Orchestrator
Replicates the LangGraph planner → actor → evaluator loop
via Bedrock AgentCore runtime invocations. Uses LangGraph for
standardized orchestration, but no LLM calls.
"""
import json
import os
import sys
import logging
import time
import uuid
from typing import Any, Optional

import boto3
from botocore.config import Config
from bedrock_agentcore.runtime import BedrockAgentCoreApp

from langgraph.graph import StateGraph, START, END

# ==================== LOGGING ====================

metric_logger = logging.getLogger("agent_metrics")
metric_logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(message)s'))
metric_logger.addHandler(handler)

from datetime import datetime, timezone


# ==================== SUB-AGENT INVOCATION ====================

_client = None

def _get_client():
    global _client
    if _client is None:
        config = Config(read_timeout=900)
        _client = boto3.client("bedrock-agentcore", config=config)
    return _client


def invoke_sub_agent(runtime_arn: str, payload: dict, agent_name: str) -> dict:
    """
    Call a sub-agent runtime via Bedrock AgentCore and return the parsed response.
    """
    client = _get_client()
    session_id = payload.get("session_id", "default")
    trace_id = payload.get("trace_id", uuid.uuid4().hex)

    start_time = time.time()

    try:
        response = client.invoke_agent_runtime(
            agentRuntimeArn=runtime_arn,
            qualifier="DEFAULT",
            contentType="application/json",
            accept="application/json",
            runtimeSessionId=session_id,
            payload=json.dumps(payload).encode()
        )

        # Parse response body
        body = response.get("response") or response.get("body")
        if hasattr(body, "read"):
            raw = body.read()
            text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        elif hasattr(body, '__iter__') and not isinstance(body, (str, bytes, dict)):
            parts = []
            for event in body:
                if isinstance(event, dict):
                    for v in event.values():
                        if isinstance(v, dict) and "bytes" in v:
                            parts.append(v["bytes"].decode("utf-8", errors="replace"))
                        elif isinstance(v, bytes):
                            parts.append(v.decode("utf-8", errors="replace"))
            text = "".join(parts)
        elif isinstance(body, bytes):
            text = body.decode("utf-8")
        elif isinstance(body, str):
            text = body
        else:
            text = str(body)

        latency_ms = (time.time() - start_time) * 1000

        # Try to parse as JSON
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, AttributeError):
            parsed = {"response": text}

        metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "sub_agent_call",
            "session_id": session_id,
            "trace_id": trace_id,
            "state_id": payload.get("state_id", "unknown"),
            "agent_name": agent_name,
            "latency_ms": round(latency_ms, 2),
            "status": "success",
            "response_length": len(text),
        }))

        return parsed

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "sub_agent_call",
            "session_id": session_id,
            "trace_id": trace_id,
            "state_id": payload.get("state_id", "unknown"),
            "agent_name": agent_name,
            "latency_ms": round(latency_ms, 2),
            "status": "error",
            "error": str(e),
        }))
        raise


# ==================== LANGGRAPH ORCHESTRATOR ====================

from typing import TypedDict

class OrchestratorState(TypedDict):
    prompt: str
    session_id: str
    trace_id: str
    state_id: str
    actor_id: str
    memory_config: str
    agent_state: dict          # {"iteration_count": N, "step_count": N}
    plan: Optional[str]
    actor_result: Optional[str]
    evaluation: Optional[dict]  # {"success", "needs_retry", "reason", "feedback"}


def build_orchestrator():
    planner_arn = os.environ["PLANNER_RUNTIME_ARN"]
    actor_arn = os.environ["ACTOR_RUNTIME_ARN"]
    evaluator_arn = os.environ["EVALUATOR_RUNTIME_ARN"]

    def invoke_planner(state: OrchestratorState):
        agent_state = state["agent_state"]
        feedback = None
        previous_plan = state.get("plan")
        
        evaluation = state.get("evaluation")
        if evaluation and evaluation.get("feedback"):
            feedback = evaluation["feedback"]

        planner_payload = {
            "prompt": state["prompt"],
            "session_id": state["session_id"],
            "trace_id": state["trace_id"],
            "state_id": state["state_id"],
            "actor_id": state["actor_id"],
            "memory_config": state["memory_config"],
            "agent_state": agent_state,
            "feedback": feedback,
            "previous_plan": previous_plan,
        }

        result = invoke_sub_agent(planner_arn, planner_payload, "planner")
        plan = result.get("response", str(result))
        updated_agent_state = result.get("agent_state", agent_state)

        metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "orchestrator_node",
            "node_name": "invoke_planner",
            "session_id": state["session_id"],
            "trace_id": state["trace_id"],
            "state_id": state["state_id"],
            "agent_state": updated_agent_state,
        }))

        return {"plan": plan, "agent_state": updated_agent_state}

    def invoke_actor(state: OrchestratorState):
        agent_state = state["agent_state"]

        actor_payload = {
            "plan": state["plan"],
            "prompt": state["prompt"],
            "session_id": state["session_id"],
            "trace_id": state["trace_id"],
            "state_id": state["state_id"],
            "actor_id": state["actor_id"],
            "memory_config": state["memory_config"],
            "agent_state": agent_state,
        }

        result = invoke_sub_agent(actor_arn, actor_payload, "actor")
        actor_result = result.get("response", str(result))
        updated_agent_state = result.get("agent_state", agent_state)

        metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "orchestrator_node",
            "node_name": "invoke_actor",
            "session_id": state["session_id"],
            "trace_id": state["trace_id"],
            "state_id": state["state_id"],
            "agent_state": updated_agent_state,
        }))

        return {"actor_result": actor_result, "agent_state": updated_agent_state}

    def invoke_evaluator(state: OrchestratorState):
        agent_state = state["agent_state"]

        eval_payload = {
            "original_task": state["prompt"],
            "plan": state["plan"],
            "execution": state["actor_result"],
            "session_id": state["session_id"],
            "trace_id": state["trace_id"],
            "state_id": state["state_id"],
            "actor_id": state["actor_id"],
            "memory_config": state["memory_config"],
            "agent_state": agent_state,
        }

        result = invoke_sub_agent(evaluator_arn, eval_payload, "evaluator")
        
        # Parse evaluation response
        eval_response = result.get("response", str(result))
        try:
            if isinstance(eval_response, str):
                evaluation = json.loads(eval_response)
            else:
                evaluation = eval_response
        except (json.JSONDecodeError, AttributeError):
            evaluation = {"success": True, "needs_retry": False, "reason": eval_response, "feedback": ""}

        updated_agent_state = result.get("agent_state", agent_state)

        metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "orchestrator_node",
            "node_name": "invoke_evaluator",
            "session_id": state["session_id"],
            "trace_id": state["trace_id"],
            "state_id": state["state_id"],
            "evaluation": evaluation,
            "agent_state": updated_agent_state,
        }))

        return {"evaluation": evaluation, "agent_state": updated_agent_state}

    def route_evaluator(state: OrchestratorState):
        evaluation = state.get("evaluation", {})
        agent_state = state.get("agent_state", {})
        iteration_count = agent_state.get("iteration_count", 1)
        max_retries = int(os.environ.get("MAX_RETRIES", "3"))

        if evaluation.get("needs_retry") and iteration_count < max_retries:
            return "invoke_planner"
        return END

    graph = StateGraph(OrchestratorState)
    graph.add_node("invoke_planner", invoke_planner)
    graph.add_node("invoke_actor", invoke_actor)
    graph.add_node("invoke_evaluator", invoke_evaluator)

    graph.add_edge(START, "invoke_planner")
    graph.add_edge("invoke_planner", "invoke_actor")
    graph.add_edge("invoke_actor", "invoke_evaluator")
    graph.add_conditional_edges("invoke_evaluator", route_evaluator)

    return graph.compile()


# ==================== ENTRYPOINT ====================

_orchestrator = None

def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = build_orchestrator()
    return _orchestrator


app = BedrockAgentCoreApp()

@app.entrypoint
def handle(payload):
    prompt = payload.get("prompt", "Hello!")
    session_id = payload.get("session_id", str(uuid.uuid4()))
    trace_id = payload.get("trace_id", uuid.uuid4().hex)
    actor_id = payload.get("actor_id", "default_actor_id")
    memory_config = payload.get("memory_config", "empty")
    iteration_count = payload.get("iteration_count", 0)
    
    state_id = str(uuid.uuid4())

    metric_logger.info(json.dumps({
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
        "event_type": "orchestrator_start",
        "session_id": session_id,
        "trace_id": trace_id,
        "state_id": state_id,
        "memory_config": memory_config,
        "prompt_length": len(prompt),
    }))

    orchestrator = _get_orchestrator()

    result = orchestrator.invoke({
        "prompt": prompt,
        "session_id": session_id,
        "trace_id": trace_id,
        "state_id": state_id,
        "actor_id": actor_id,
        "memory_config": memory_config,
        "agent_state": {"iteration_count": iteration_count, "step_count": 0},
        "plan": None,
        "actor_result": None,
        "evaluation": None,
    })

    evaluation = result.get("evaluation", {})
    agent_state = result.get("agent_state", {})

    return {
        "response": result.get("actor_result", ""),
        "success": evaluation.get("success", True),
        "needs_retry": evaluation.get("needs_retry", False),
        "reason": evaluation.get("reason", ""),
        "feedback": evaluation.get("feedback", None),
        "iteration_count": agent_state.get("iteration_count", 1),
        "max_iterations": int(os.environ.get("MAX_RETRIES", "3")),
        "state_id": state_id,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/Users/haseeb/Code/iisc/bedrockAC/workflows/react_distributed/react_orchestrator/.env.dev")
    app.run()