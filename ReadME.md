# Bedrock AgentCore Deployment Guide

This guide covers how to deploy agent workflows using the Bedrock AgentCore runtime, specifically when you need to circumvent AWS CodeBuild policy requirements by performing a custom local build via Docker.

## 1. Create the App Entry Point

The `app.py` script acts as the main entry point for the Bedrock AgentCore Runtime. It must define a `@workflow` or `@task` function annotated with `@app.entrypoint`.

Example structure:
```python
import os
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from traceloop.sdk.decorators import workflow

app = BedrockAgentCoreApp()

@app.entrypoint
@workflow(name="mcp_workflow")
def handle(payload):
    # Retrieve user input, session ID, etc.
    prompt = payload.get("prompt", "Hello!")
    
    # Check for needed environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY not set"}
    
    # Process the request using your LLM/Agent graph
    result = {"response": "Processed output"}
    
    return result

if __name__ == "__main__":
    app.run()
```

## 2. Create a Custom Dockerfile

To bypass AWS CodeBuild requirements and directly deploy the Docker image, create a `Dockerfile` that packages your application logic, dependencies, and the `app.py` entry point.

Example `Dockerfile`:

```dockerfile
FROM python:3.12-slim

# Install uv from the official Astral image for fast package installations
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# (Optional) Copy custom packages, for instance:
# COPY custom_mcp ./custom_mcp

# Copy requirements
COPY requirements.txt .

# Install dependencies using uv
RUN uv pip install --system --no-cache -r requirements.txt 
# If utilizing local packages: RUN uv pip install --system --no-cache -r requirements.txt ./custom_mcp

# Copy core agent code and prompts
COPY app.py .
# COPY other_files.py .

EXPOSE 8080

CMD ["opentelemetry-instrument", "python", "app.py"]
```

> **Note**: This Dockerfile specifies the custom entry point command to run the runtime properly and instrument it with OpenTelemetry.

## 3. Configure the Deployment (`agentcore configure`)

When you run `agentcore configure`, specify the path to `app.py` as the entry point and choose `container` as the deployment method. This will generate a `.bedrock_agentcore.yaml` configuration file. 
You can choose to **not** auto-create ECR and CodeBuild policies if you prefer handling the deployment manually.

## 4. Include Environment Variables and IAM Permissions

Environment variables (like `OPENAI_API_KEY` or `MODEL_NAME`) are not automatically packaged into the image unless hardcoded (not recommended). You should add them via AWS Lambda/ECS execution environments.

```bash
export $(grep OPENAI_API_KEY /Users/haseeb/Code/iisc/bedrockAC/.env | xargs)
```

Ensure that the IAM execution role you use for the AgentCore runtime has permissions to fetch these configuration parameters if using Systems Manager (SSM) or Secrets Manager, or pass them in directly through the container configuration.

> **Important**: Ensure your IAM execution role has proper permissions if your agent requires them. For example, if you interact with external Lambda functions via URL (e.g., a remote MCP server), the execution role must have `lambda:GetFunctionUrlConfig` and `lambda:InvokeFunctionUrl` permissions.

## 5. Deployment and Invocation

1. Build and push your Docker image to an ECR repository.

```bash
export $(grep OPENAI_API_KEY /Users/haseeb/Code/iisc/bedrockAC/.env | xargs)
agentcore deploy --local-build --env OPENAI_API_KEY=$OPENAI_API_KEY
```

2. Ensure the execution IAM role attached to the AgentCore Service has the proper policies.
3. Once the Bedrock AgentCore runtime is active, you can invoke it using 

```bash
agentcore invoke '{"prompt": "Get and sumarize the introduction and core contributions of the paper - Multi-scale competition in the Majorana-Kondo system"}'
```

`boto3`:

```python
import sys
import boto3
import json

REGION = "ap-south-1"
RUNTIME_ARN = "arn:aws:bedrock-agentcore:ap-south-1:<YOUR-ACCOUNT-ID>:runtime/<YOUR-AGENT-ID>"
ENDPOINT_NAME = "DEFAULT"

def invoke_agent(prompt: str):
    client = boto3.client("bedrock-agentcore", region_name=REGION)

    print(f"🔗 Invoking AgentCore Runtime...")
    
    response = client.invoke_agent_runtime(
        agentRuntimeArn=RUNTIME_ARN,
        qualifier=ENDPOINT_NAME,
        contentType="application/json",
        accept="application/json",
        payload=json.dumps({"prompt": prompt}).encode(),
    )

    body = json.loads(response["response"].read().decode())
    print(body.get('response', body))

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello!"
    invoke_agent(prompt)
```

---

## 📖 Reference: Custom local-build MCP Monolith (`plan_act_evaluate_mcp_monolith`)

As a practical example, to deploy the `plan_act_evaluate_mcp_monolith` workflow, you would reference its local package structure:

**Directory Structure**:
```
workflows/plan_act_evaluate_mcp_monolith/
├── Dockerfile
├── requirements.txt
├── app.py
├── prompts.py
└── invoke.py
```

**Workflow `Dockerfile` Example**:
```dockerfile
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app

COPY custom_mcp ./custom_mcp
COPY workflows/plan_act_evaluate_mcp_monolith/requirements.txt .

RUN uv pip install --system --no-cache -r requirements.txt ./custom_mcp

COPY workflows/plan_act_evaluate_mcp_monolith/app.py .
COPY workflows/plan_act_evaluate_mcp_monolith/prompts.py .

EXPOSE 8080
CMD ["opentelemetry-instrument", "python", "app.py"]
```
*(Ensure you build from the root directory so `custom_mcp` and `workflows` paths are valid.)*
