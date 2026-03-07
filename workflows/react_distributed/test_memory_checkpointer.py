"""
Test module for debugging AgentCoreMemorySaver behavior across different LangGraph graphs.

Simulates the distributed ReAct pattern:
  - Graph A (like planner+actor): 2 nodes, writes messages to checkpoint
  - Graph B (like evaluator): 2 nodes, loads checkpoint and checks what messages it sees

Both graphs share the same MEMORY_ID and thread_id, mimicking the distributed setup.
"""

import json
import os
import uuid
from typing import Optional

from dotenv import load_dotenv

# Load env from the evaluator's .env.dev (has MEMORY_ID, OPENAI_API_KEY, AWS_REGION)
load_dotenv("/Users/haseeb/Code/iisc/bedrockAC/workflows/react_distributed/evaluator/.env.dev")

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph_checkpoint_aws import AgentCoreMemorySaver


# ==================== CONFIG ====================

MEMORY_ID = os.environ["MEMORY_ID"]
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
THREAD_ID = f"test-debug-{uuid.uuid4().hex[:8]}"
ACTOR_ID = "test-actor"

print(f"=== Memory Checkpointer Debug Test ===")
print(f"MEMORY_ID: {MEMORY_ID}")
print(f"AWS_REGION: {AWS_REGION}")
print(f"THREAD_ID: {THREAD_ID}")
print(f"ACTOR_ID:  {ACTOR_ID}")
print()


# ==================== SHARED CHECKPOINTER ====================

checkpointer = AgentCoreMemorySaver(MEMORY_ID, region_name=AWS_REGION)

CONFIG = {
    "configurable": {
        "thread_id": THREAD_ID,
        "actor_id": ACTOR_ID,
    }
}


# ==================== GRAPH A: "Planner-Actor" (2 nodes) ====================

class GraphAState(MessagesState):
    plan: Optional[str]
    step_count: Optional[int]


def build_graph_a():
    """Graph A simulates planner → actor. Two nodes, operating on messages."""

    def node_planner(state: GraphAState):
        messages = state["messages"]
        step = (state.get("step_count") or 0) + 1

        print(f"  [Graph A / node_planner] step={step}")
        print(f"  [Graph A / node_planner] incoming messages count: {len(messages)}")
        for i, m in enumerate(messages):
            print(f"    msg[{i}]: type={m.type}, content={str(m.content)[:100]}")

        # Simulate planner response
        response = AIMessage(content="PLAN: 1) Use document_retriever to fetch the paper. 2) Summarize key findings.")
        return {
            "messages": [response],
            "plan": response.content,
            "step_count": step,
        }

    def node_actor(state: GraphAState):
        messages = state["messages"]
        step = (state.get("step_count") or 0) + 1

        print(f"  [Graph A / node_actor] step={step}")
        print(f"  [Graph A / node_actor] incoming messages count: {len(messages)}")
        for i, m in enumerate(messages):
            print(f"    msg[{i}]: type={m.type}, content={str(m.content)[:100]}")

        # Simulate actor doing work + tool call + final response
        response = AIMessage(content="ACTOR RESULT: The paper discusses quantum computing advances. Key finding: 40% speedup over classical.")
        return {
            "messages": [response],
            "step_count": step,
        }

    graph = StateGraph(GraphAState)
    graph.add_node("planner", node_planner)
    graph.add_node("actor", node_actor)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "actor")
    graph.add_edge("actor", END)

    return graph.compile(checkpointer=checkpointer)


# ==================== GRAPH B: "Evaluator" (2 nodes) ====================

class GraphBState(MessagesState):
    original_task: Optional[str]
    evaluation: Optional[dict]
    step_count: Optional[int]


def build_graph_b():
    """Graph B simulates evaluator. Two nodes — preprocessor + evaluator."""

    def node_preprocessor(state: GraphBState):
        messages = state["messages"]
        step = (state.get("step_count") or 0) + 1

        print(f"  [Graph B / node_preprocessor] step={step}")
        print(f"  [Graph B / node_preprocessor] incoming messages count: {len(messages)}")
        for i, m in enumerate(messages):
            print(f"    msg[{i}]: type={m.type}, content={str(m.content)[:100]}")

        # Preprocessor doesn't add messages, just passes through
        return {"step_count": step}

    def node_evaluator(state: GraphBState):
        messages = state["messages"]
        step = (state.get("step_count") or 0) + 1

        print(f"  [Graph B / node_evaluator] step={step}")
        print(f"  [Graph B / node_evaluator] incoming messages count: {len(messages)}")
        for i, m in enumerate(messages):
            print(f"    msg[{i}]: type={m.type}, content={str(m.content)[:100]}")

        evaluation = {"success": True, "needs_retry": False, "reason": "Looks good"}
        return {
            "evaluation": evaluation,
            "step_count": step,
        }

    graph = StateGraph(GraphBState)
    graph.add_node("preprocessor", node_preprocessor)
    graph.add_node("evaluator", node_evaluator)
    graph.add_edge(START, "preprocessor")
    graph.add_edge("preprocessor", "evaluator")
    graph.add_edge("evaluator", END)

    return graph.compile(checkpointer=checkpointer)


# ==================== DEBUG: Read checkpoint directly ====================

def debug_read_checkpoint(label: str):
    """Read and print the current checkpoint state for our thread_id."""
    print(f"\n--- DEBUG CHECKPOINT: {label} ---")
    saved = checkpointer.get_tuple(CONFIG)
    if saved is None:
        print("  (no checkpoint found)")
        return

    checkpoint = saved.checkpoint
    channel_values = checkpoint.get("channel_values", {})
    channel_versions = checkpoint.get("channel_versions", {})
    versions_seen = checkpoint.get("versions_seen", {})

    print(f"  checkpoint_id: {checkpoint.get('id')}")
    print(f"  metadata step: {saved.metadata.get('step')}")
    print(f"  metadata source: {saved.metadata.get('source')}")
    print(f"  channel_versions keys: {list(channel_versions.keys())}")
    print(f"  versions_seen keys: {list(versions_seen.keys())}")
    print(f"  channel_values keys: {list(channel_values.keys())}")

    if "messages" in channel_values:
        msgs = channel_values["messages"]
        print(f"  messages count: {len(msgs)}")
        for i, m in enumerate(msgs):
            content_preview = str(m.content)[:120] if hasattr(m, 'content') else str(m)[:120]
            msg_type = m.type if hasattr(m, 'type') else type(m).__name__
            print(f"    [{i}] type={msg_type}, content={content_preview}")
    else:
        print("  messages: NOT PRESENT in channel_values")

    # Print other channels
    for k, v in channel_values.items():
        if k != "messages":
            print(f"  channel '{k}': {str(v)[:200]}")
    print(f"--- END CHECKPOINT ---\n")


# ==================== MAIN TEST ====================

def main():
    graph_a = build_graph_a()
    graph_b = build_graph_b()

    # ---- Step 0: Check initial checkpoint state ----
    debug_read_checkpoint("Before anything")

    # ---- Step 1: Run Graph A (planner-actor) ----
    print("=" * 60)
    print("STEP 1: Running Graph A (planner → actor)")
    print("=" * 60)

    user_query = "Summarize the key contributions of the paper on quantum computing advances."

    result_a = graph_a.invoke(
        {
            "messages": [HumanMessage(content=user_query)],
            "plan": None,
            "step_count": 0,
        },
        config=CONFIG,
    )

    print(f"\nGraph A result keys: {list(result_a.keys())}")
    print(f"Graph A final messages count: {len(result_a.get('messages', []))}")
    for i, m in enumerate(result_a.get("messages", [])):
        print(f"  [{i}] type={m.type}, content={str(m.content)[:100]}")

    # ---- Step 2: Check checkpoint after Graph A ----
    debug_read_checkpoint("After Graph A")

    # ---- Step 3: Run Graph B (evaluator) with its own input ----
    print("=" * 60)
    print("STEP 2: Running Graph B (preprocessor → evaluator)")
    print("  Graph B uses the SAME thread_id and MEMORY_ID.")
    print("  If the checkpointer shares state, Graph B should see")
    print("  Graph A's messages in its state.")
    print("=" * 60)

    result_b = graph_b.invoke(
        {
            "messages": [HumanMessage(content=user_query)],
            "original_task": user_query,
            "evaluation": None,
            "step_count": 0,
        },
        config=CONFIG,
    )

    print(f"\nGraph B result keys: {list(result_b.keys())}")
    print(f"Graph B final messages count: {len(result_b.get('messages', []))}")
    for i, m in enumerate(result_b.get("messages", [])):
        print(f"  [{i}] type={m.type}, content={str(m.content)[:100]}")

    # ---- Step 4: Final checkpoint state ----
    debug_read_checkpoint("After Graph B")

    # ---- Summary ----
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    a_msg_count = len(result_a.get("messages", []))
    b_msg_count = len(result_b.get("messages", []))
    print(f"Graph A produced {a_msg_count} messages")
    print(f"Graph B saw {b_msg_count} messages")
    if b_msg_count > 1:
        print("✅ Graph B saw messages from Graph A via the checkpointer!")
    else:
        print("❌ Graph B did NOT see Graph A's messages — checkpointer did not share state across graphs.")

    # ---- Cleanup: Delete the test thread ----
    print(f"\nCleaning up test thread: {THREAD_ID}")
    try:
        checkpointer.delete_thread(THREAD_ID, ACTOR_ID)
        print("Cleanup done.")
    except Exception as e:
        print(f"Cleanup failed (non-critical): {e}")


if __name__ == "__main__":
    main()
