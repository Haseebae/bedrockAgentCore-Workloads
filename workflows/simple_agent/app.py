"""
Minimal AgentCore agent — answers questions using LangGraph + OpenAI.
"""
import os
from langchain_core.messages import HumanMessage
from typing import Optional
from langgraph.graph import StateGraph, START, MessagesState
from langchain.chat_models import init_chat_model
from bedrock_agentcore.runtime import BedrockAgentCoreApp
# from dotenv import load_dotenv  

# load_dotenv("/Users/haseeb/Code/iisc/bedrockAgentCore/.env")

class AgentState(MessagesState):
    model_name: Optional[str]

# Define the graph
def call_model(state: AgentState):
    model_name = state.get("model_name") or "openai:gpt-4o-mini"
    model = init_chat_model(model_name)
    response = model.invoke(state["messages"])
    return {"messages": response}

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_edge(START, "agent")
graph = workflow.compile()

app = BedrockAgentCoreApp()

@app.entrypoint
def handle(payload):
    """
    payload: {"prompt": "What is cloud computing?"}
    returns: {"response": "Cloud computing is..."}
    """
    prompt = payload.get("prompt", "Hello!")
    
    # Ensure OPENAI_API_KEY is robustly handled or passed
    if not os.environ.get("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY not set"}

    # Run the graph
    result = graph.invoke({"messages": [HumanMessage(content=prompt)]})
    
    # Extract the response
    msg = result["messages"][-1]
    return {"response": msg.content}


if __name__ == "__main__":
    app.run()
