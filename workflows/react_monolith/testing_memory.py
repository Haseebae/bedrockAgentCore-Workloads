import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Optional, Any
from langgraph.graph import StateGraph, START, END, MessagesState

# Import the AgentCore Memory integrations
from langgraph_checkpoint_aws import AgentCoreMemorySaver
from dotenv import load_dotenv

load_dotenv("/Users/haseeb/Code/iisc/bedrockAC/workflows/react_monolith/.env.dev")

REGION = os.environ.get("REGION", "ap-south-1")
MEMORY_ID = os.environ.get("MEMORY_ID", "YOUR_MEMORY_ID")

checkpointer = AgentCoreMemorySaver(MEMORY_ID, region_name=REGION)

model_name = os.environ.get("MODEL_NAME", "openai:gpt-4o-mini")
model = init_chat_model(model_name)

class AgentState(MessagesState):
    plan: Optional[str]
    iteration_count: Optional[int]

def planner_node(state: AgentState):
    print("\n--- [PLANNER NODE EXECUTING] ---")
    messages = state.get("messages", [])
    iteration_count = state.get("iteration_count", 0)
    print(f"State -> iteration_count: {iteration_count}")
    print(f"State -> messages count: {len(messages)}")
    for m in messages:
        print(f"  [{m.__class__.__name__}]: {m.content[:50]}")
    
    # Simulate planning
    plan = f"Plan for iteration {iteration_count + 1}"
    print(f"Planner creates plan: '{plan}'")
    
    return {"plan": plan, "iteration_count": iteration_count + 1}

def actor_node(state: AgentState):
    print("\n--- [ACTOR NODE EXECUTING] ---")
    messages = state.get("messages", [])
    plan = state.get("plan", "No plan")
    iteration_count = state.get("iteration_count", 0)
    print(f"State -> iteration_count: {iteration_count}")
    print(f"State -> plan: {plan}")
    print(f"State -> messages count: {len(messages)}")
    
    # Generate response
    sys_msg = SystemMessage(content=f"You are an actor following this plan: {plan}. Be brief.")
    response = model.invoke([sys_msg] + messages)
    print(f"Actor generates response: {response.content[:50]}...")
    
    return {"messages": [response]}

def evaluator_node(state: AgentState):
    print("\n--- [EVALUATOR NODE EXECUTING] ---")
    messages = state.get("messages", [])
    print(f"State -> messages count: {len(messages)}")
    
    # Doesn't modify messages, just prints
    last_msg = messages[-1] if messages else None
    print(f"Evaluator sees last message: {last_msg.content[:50]}..." if last_msg else "No messages")
    
    return {}

graph = StateGraph(AgentState)
graph.add_node("planner", planner_node)
graph.add_node("actor", actor_node)
graph.add_node("evaluator", evaluator_node)

graph.add_edge(START, "planner")
graph.add_edge("planner", "actor")
graph.add_edge("actor", "evaluator")
graph.add_edge("evaluator", END)

app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "test_thread_mimic_1", "actor_id": "test_actor_1", "session_id": "test_session_1"}}

print("\n=======================================================")
print("=== MULTI-TURN SIMULATION WITH LANGGRAPH NODES      ===")
print("=======================================================")

print("\n\n>>> INITIAL TURN: User says hello and gives context")
initial_state = {
    "messages": [HumanMessage(content="Hi, my name is Haseeb and my favorite fruit is apple.")],
    "iteration_count": 0,
    "plan": None
}

for event in app.stream(initial_state, config=config, stream_mode="updates"):
    pass

print("\n\n>>> SECOND TURN: User asks a follow-up question")
# Notice we ONLY provide the new message. The checkpointer is responsible for injecting the history.
followup_state = {
    "messages": [HumanMessage(content="Can you recall my name and favorite fruit?")]
}

for event in app.stream(followup_state, config=config, stream_mode="updates"):
    pass

print("\n\n>>> THIRD TURN: User asks a follow-up question")
# Notice we ONLY provide the new message. The checkpointer is responsible for injecting the history.
followup_state = {
    "messages": [HumanMessage(content="What is my name and what fruit do I like?")]
}

config = {"configurable": {"thread_id": "test_thread_mimic_new", "actor_id": "test_actor_new", "session_id": "test_session_new"}}

for event in app.stream(followup_state, config=config, stream_mode="updates"):
    pass

print("\n\n=======================================================")
print("=== FINAL STATE PULLED FROM CHECKPOINTER            ===")
print("=======================================================")
state = app.get_state(config)
print(f"Final iteration count: {state.values.get('iteration_count')}")
print(f"Final plan: {state.values.get('plan')}")
print("Final Messages:")
for idx, m in enumerate(state.values.get("messages", [])):
    print(f" {idx+1}. [{m.__class__.__name__}]: {m.content}")
