from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
import contextvars
import logging
import json
import time
from datetime import datetime, timezone


class SessionMetricsCallback(BaseCallbackHandler):
    def __init__(
        self, 
        session_id: str, 
        current_node_var: contextvars.ContextVar, 
        metric_logger: logging.Logger,
        trace_id_var: contextvars.ContextVar = None,
        state_id_var: contextvars.ContextVar = None
    ):
        self.session_id = session_id
        self.llm_starts = {}
        self.current_node_var = current_node_var
        self.metric_logger = metric_logger
        self.trace_id_var = trace_id_var
        self.state_id_var = state_id_var
    
    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id, **kwargs):
        node_name = self.current_node_var.get()
        
        # Calculate input bytes from the serialized prompt strings
        input_bytes = sum(len(p.encode("utf-8")) for p in prompts) if prompts else 0
        
        self.llm_starts[run_id] = {
            "start": time.time(),
            "node_name": node_name,
            "input_bytes": input_bytes
        }

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id, **kwargs):
        node_name = self.current_node_var.get()
        
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
                        
        self.llm_starts[run_id] = {
            "start": time.time(),
            "node_name": node_name,
            "input_bytes": input_bytes
        }

    def on_llm_end(self, response: LLMResult, *, run_id, parent_run_id, **kwargs):
        latency = 0
        node_name = "unknown"
        input_bytes = 0
        if run_id in self.llm_starts:
            start_info = self.llm_starts.pop(run_id)
            latency = (time.time() - start_info["start"]) * 1000
            node_name = start_info["node_name"]
            input_bytes = start_info["input_bytes"]

        input_tokens = 0
        output_tokens = 0
        output_bytes = 0
        
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
        elif response.generations and hasattr(response.generations[0][0], "message"):
            msg = response.generations[0][0].message
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                input_tokens = msg.usage_metadata.get("input_tokens", 0)
                output_tokens = msg.usage_metadata.get("output_tokens", 0)

        # Calculate output bytes from generation text
        if response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, "message"):
                        msg = gen.message
                        
                        # Add content bytes
                        if hasattr(msg, "content") and msg.content:
                            if isinstance(msg.content, str):
                                output_bytes += len(msg.content.encode("utf-8"))
                            else:
                                output_bytes += len(json.dumps(msg.content).encode("utf-8"))
                        
                        # Add tool_calls bytes
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            output_bytes += len(json.dumps(msg.tool_calls).encode("utf-8"))
                        elif hasattr(msg, "additional_kwargs") and msg.additional_kwargs.get("tool_calls"):
                            output_bytes += len(json.dumps(msg.additional_kwargs["tool_calls"]).encode("utf-8"))
                            
                    elif hasattr(gen, "text"):
                        if isinstance(gen.text, str):
                            output_bytes += len(gen.text.encode("utf-8"))
                        else:
                            output_bytes += len(json.dumps(gen.text).encode("utf-8"))

        trace_id = self.trace_id_var.get() if self.trace_id_var else "unknown_trace"
        state_id = self.state_id_var.get() if self.state_id_var else "unknown_state"

        self.metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "llm_call",
            "session_id": self.session_id,
            "trace_id": trace_id,
            "state_id": state_id,
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "node_name": node_name,
            "latency_ms": round(latency, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes
        }))

