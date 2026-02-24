# AgentCore Quickstart: Detailed Implementation Log

**Date:** 2026-02-17T21:56 IST
**Status:** ✅ SUCCESS — Simple Agent Deployed & Invocable

This document logs the exact steps, services, and technical details used to successfully deploy a "Hello World" agent to Amazon Bedrock AgentCore Runtime.

---

## 1. Architecture & Concepts

### The Wrapper: `BedrockAgentCoreApp`
This acts as an **Interface Adapter**. It translates AgentCore Runtime requests into standard Python objects for your code.
- **Input/Output Contract**: Fully defined by **your code**.
  - We chose input: `{"prompt": "..."}`
  - We chose output: `{"response": "..."}`
  - *Standardization*: None enforced by AgentCore. You can use any JSON schema.
- **The "Black Box"**: Inside the `@app.entrypoint`, you can run **anything**:
  - LangGraph workflows
  - Local models (HuggingFace/vLLM inside the container)
  - External API calls (OpenAI, Anthropic)
  - Python logic
- **Endpoints**: The wrapper exposes `/invocations` (POST) for requests and `/ping` (GET) for health checks.

### Infrastructure: Runtime vs. Endpoint
- **Runtime**: The compute resource (Container + IAM + Config). Think of it as the "backend service".
- **Runtime Endpoint**: The access door.
  - Decoupled from the runtime to allow safe updates (e.g., swapping container images behind a stable URL).
  - Can have multiple endpoints provided for one runtime (Resulted in `default` user-created and `DEFAULT` auto-created endpoints).

---

## 2. Prerequisites & Setup

### Tools & CLIs
- **Python**: 3.12.10 (via Homebrew/uv)
- **AWS CLI**: Configured with `bedrock-agentcore` permissions
- **Docker**: Engine 28.4.0 (for building agent container)
- **uv**: Used for fast virtual environment management

### SDK Installation
We used a virtual environment with the following keys packages:
- `bedrock-agentcore` (v0.0.1 - Internal SDK for AgentCore)
- `strands-agents` (v1.26.0 - Agent framework)
- `boto3` (AWS SDK)

**Command:**
```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install bedrock-agentcore strands-agents boto3
```

---

## 3. Agent Development (`simple_agent/app.py`)

We created a minimal agent using the `BedrockAgentCoreApp` wrapper.

**Key Implementation Details:**
- **Entrypoint**: Used `@app.entrypoint` decorator to expose the handler.
- **Model Config**: Switched from `ap-south-1` to `us-east-1` because `amazon.nova-lite-v1:0` supports on-demand inference directly in us-east-1 without cross-region profiles.
- **Code Structure**:
  ```python
  from bedrock_agentcore.runtime import BedrockAgentCoreApp
  from strands import Agent
  from strands.models.bedrock import BedrockModel

  app = BedrockAgentCoreApp()

  @app.entrypoint
  def handle(payload):
      # ... model initialization & invocation ...
      return {"response": str(result)}
  ```

**Streaming Support**:
To support streaming, change `return` to `yield`. The wrapper automatically handles the streaming response protocol.
```python
@app.entrypoint
def handle(payload):
    yield {"status": "started"}
    # ... logic ...
    yield {"result": "partial..."}
```

---

## 4. Deployment Pipeline (`deploy_simple.py`)

We built a custom Python script to handle the full deployment lifecycle using `boto3`.

### AWS Services Interacted With:
1.  **IAM**: Created execution role `BedrockAgentCoreExecutionRole`.
    *   **Trust Policy**: Allowed `bedrock-agentcore.amazonaws.com`.
    *   **Permissions**: Attached `AmazonBedrockFullAccess` (model invocation) and `AmazonEC2ContainerRegistryReadOnly` (pulling container).
2.  **Amazon ECR**: Created repository `bedrock-agentcore/simple-agent` and pushed the Docker image.
3.  **Bedrock AgentCore Control Plane** (`bedrock-agentcore-control`):
    *   Managed the Runtime and Endpoints.

### Deployment Steps (Automated):
1.  **Build**: `docker build -t ...`
2.  **Push**: `docker push ...` to ECR.
3.  **Create Runtime**: Called `create_agent_runtime` with `containerConfiguration`.
    *   **Constraint Discovery**: Runtime name must be `[a-zA-Z][a-zA-Z0-9_]{0,47}` (No hyphens allowed). We renamed `simple-hello-agent` → `simple_hello_agent`.
4.  **Create Endpoint**: Called `create_agent_runtime_endpoint`.
5.  **Wait**: Polled until status reached `READY`.

---

## 5. The Critical Fix: Invocation Strategy

This was the main technical hurdle. We attempted three approaches before finding the working solution.

### ❌ Attempt 1: WebSocket with `websockets` Library
- **Method**: Used SDK to generate SigV4 signed headers (`generate_ws_connection`).
- **Result**: `HTTP 424 (Failed Dependency)`.
- **Cause**: Header conflict between SDK-generated headers (`Upgrade`, `Connection`) and those managed by the `websockets` library.

### ❌ Attempt 2: Presigned WebSocket URL
- **Method**: Used `generate_presigned_url` to embed auth in the query string.
- **Result**: `HTTP 424` with body `Received error (403) from runtime`.
- **Cause**: The runtime container (or the gateway's connection to it) was rejecting the connection, possibly due to cold start timing or specific WebSocket handshake requirements that proved brittle.

### ✅ Attempt 3: Data Plane API (`invoke_agent_runtime`)
- **Method**: We discovered the `bedrock-agentcore` (no suffix) service in boto3 is the **Data Plane**.
- **Action**: Used `invoke_agent_runtime` (Ref: `InvokeAgentRuntime` operation).
- **Result**: **Success (HTTP 200)**.
- **Why it worked**: This is the standard AWS API invocation path. It handles authentication, routing, and session management natively without managing raw WebSocket connections.

**Final Working Invocation Code (`invoke_simple.py`):**
```python
client = boto3.client("bedrock-agentcore", region_name="us-east-1")
response = client.invoke_agent_runtime(
    agentRuntimeArn=RUNTIME_ARN,
    qualifier="default",
    contentType="application/json",
    accept="application/json",
    payload=json.dumps({"prompt": prompt}).encode(),
)
# Read response['response'] streaming body to get final result
```

---

## Summary of Resources Created

| Resource | Service | Identifier | Status |
| :--- | :--- | :--- | :--- |
| **Execution Role** | IAM | `BedrockAgentCoreExecutionRole` | Active |
| **Container Repo** | ECR | `bedrock-agentcore/simple-agent` | Active |
| **Agent Runtime** | AgentCore | `simple_hello_agent-AZewvx92jY` | **READY** |
| **Endpoint** | AgentCore | `default` | **READY** |

## Next Steps

Now that the foundational pipeline is working (Build → Push → Deploy → Invoke), we can proceed to the original objective: **Building the Generic AgentCore Client** using this same successful pattern.

---

## 6. Troubleshooting & Fixes (2026-02-18)

We encountered `RuntimeClientError` and `CloudWatch` log issues, which were traced to specific hard requirements and API nuances.

### 6.1. Hard Requirement: ARM64 Architecture
- **Error**: `ValidationException: ... Supported architectures: [arm64]`
- **Fix**: Docker images **must** be built for `linux/arm64` (Graviton). The AgentCore Runtime **does not support** `linux/amd64`.
- **Command**: `docker build --platform linux/arm64 ...`

### 6.2. API Response Implementation Details
The `boto3` client for `bedrock-agentcore-control` behaves differently than standard AWS patterns:

1.  **List Runtimes**: Returns `agentRuntimes`, NOT `agentRuntimeSummaries`.
2.  **List Endpoints**: Returns `runtimeEndpoints`, NOT `agentRuntimeEndpointSummaries`.
    *   Endpoint ID field is `id`, NOT `agentRuntimeEndpointId`.

### 6.3. Runtime Updates
- **Requirement**: `update_agent_runtime` requires all mandatory fields (`roleArn`, `networkConfiguration`), not just the fields being updated.

### 6.4. Environment Variables
- **Issue**: `load_dotenv()` with a local file path causes the container to crash.
- **Fix**: Use `environmentVariables` in `create_agent_runtime` to inject config.
