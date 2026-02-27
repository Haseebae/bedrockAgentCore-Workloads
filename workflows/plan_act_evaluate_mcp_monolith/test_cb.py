import time
import json
import contextvars
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

session_id_var = contextvars.ContextVar("session_id", default="test_session")
current_node_var = contextvars.ContextVar("current_node", default="test_node")

class SessionMetricsCallback(BaseCallbackHandler):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.llm_starts = {}
    
    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        print(f"DEBUG: on_llm_start called with prompts: {len(prompts)} prompts")
        node_name = current_node_var.get()
        input_bytes = sum(len(p.encode("utf-8")) for p in prompts) if prompts else 0
        self.llm_starts[run_id] = {
            "start": time.time(),
            "node_name": node_name,
            "input_bytes": input_bytes
        }

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        print(f"DEBUG: on_chat_model_start called with {len(messages)} messages lists")
        node_name = current_node_var.get()
        
        input_bytes = 0
        if messages:
            for msg_list in messages:
                for msg in msg_list:
                    if hasattr(msg, "content") and msg.content:
                        if isinstance(msg.content, str):
                            input_bytes += len(msg.content.encode("utf-8"))
                        else:
                            input_bytes += len(json.dumps(msg.content).encode("utf-8"))
                    
                    if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
                        input_bytes += len(json.dumps(msg.additional_kwargs).encode("utf-8"))
                    elif hasattr(msg, "tool_calls") and msg.tool_calls:
                        input_bytes += len(json.dumps(msg.tool_calls).encode("utf-8"))
        
        # Add bytes for bound tools/functions which consume significant input tokens
        invocation_params = kwargs.get("invocation_params", {})
        if "tools" in invocation_params:
            input_bytes += len(json.dumps(invocation_params["tools"]).encode("utf-8"))
        elif "functions" in invocation_params:
            input_bytes += len(json.dumps(invocation_params["functions"]).encode("utf-8"))
        
        # kwargs might contain 'tools' directly depending on the langchain version
        if "tools" in kwargs and "tools" not in invocation_params:
            input_bytes += len(json.dumps(kwargs["tools"]).encode("utf-8"))
            
        print(f"DEBUG: on_chat_model_start calculated input_bytes = {input_bytes}")
        self.llm_starts[run_id] = {
            "start": time.time(),
            "node_name": node_name,
            "input_bytes": input_bytes
        }

    def on_llm_end(self, response: LLMResult, *, run_id, parent_run_id=None, **kwargs):
        print("DEBUG: on_llm_end called")
        if run_id in self.llm_starts:
            start_info = self.llm_starts.pop(run_id)
            print(f"DEBUG: popping run_id with input_bytes = {start_info['input_bytes']}")
        else:
            print("DEBUG: run_id not found in llm_starts!")

llm = ChatOpenAI(model="gpt-4o-mini").bind_tools([{"type": "function", "function": {"name": "test", "description": "test", "parameters": {"type": "object", "properties": {}}}}])

msgs = [
    SystemMessage(content="You are a helpful assistant. " * 50),
    HumanMessage(content="Hello! " * 50)
]

cb = SessionMetricsCallback(session_id="test")
llm.invoke(msgs, config={"callbacks": [cb]})
