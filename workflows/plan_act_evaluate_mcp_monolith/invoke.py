"""
Invoke the deployed agent on AgentCore Runtime.

Usage:
    python invoke_react.py "Your question here"
    python invoke_react.py   # Uses default prompt
"""

import boto3
import json
import sys

# ─── Configuration ──────────────────────────────────────────────────────────
REGION = "ap-south-1"
RUNTIME_ARN = "arn:aws:bedrock-agentcore:ap-south-1:235319806087:runtime/reactmcp-tuBkEhG2oo"
ENDPOINT_NAME = "DEFAULT"


def invoke_agent(prompt: str) -> dict:
    """Invoke the deployed agent via the bedrock-agentcore data plane API."""

    client = boto3.client("bedrock-agentcore", region_name=REGION)

    print(f"🔗 Invoking AgentCore Runtime...")
    print(f"   Runtime: {RUNTIME_ARN.split('/')[-1]}")
    print(f"   Endpoint: {ENDPOINT_NAME}")
    print(f"\n📤 Prompt: {prompt}")

    # Generate a dummy traceparent for now just to see it work
    import uuid
    trace_id = uuid.uuid4().hex
    span_id = uuid.uuid4().hex[:16]
    traceparent = f"00-{trace_id}-{span_id}-01"
    
    session_id = str(uuid.uuid4())

    response = client.invoke_agent_runtime(
        agentRuntimeArn=RUNTIME_ARN,
        qualifier=ENDPOINT_NAME,
        contentType="application/json",
        accept="application/json",
        runtimeSessionId=session_id,
        traceId=trace_id,
        traceParent=traceparent,
        traceState=f"rojo={span_id}",
        payload=json.dumps({"prompt": prompt, "session_id": session_id}).encode(),
    )

    status = response["statusCode"]
    session_id = response.get("runtimeSessionId", "N/A")
    body = json.loads(response["response"].read().decode())

    print(f"\n📥 Status: {status}")
    print(f"   Session: {session_id}")
    print(f"\n✅ Response:")
    print(f"   {body.get('response', body)}")

    return body


if __name__ == "__main__":
    prompt = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Get and sumarize the introduction and core contributions of the paper - Multi-scale competition in the Majorana-Kondo system"
    )

    print(f"🚀 Invoking deployed agent on AgentCore Runtime\n")
    result = invoke_agent(prompt)

    print(f"\n{'='*60}")
    print(f"  🎉 Remote invocation successful!")
    print(f"{'='*60}")
