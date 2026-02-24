"""
Deploy simple agent to Bedrock AgentCore Runtime (container approach).

Steps:
  1. Create IAM execution role (if not exists)
  2. Build Docker image
  3. Push to ECR
  4. Create AgentCore Runtime with container config
  5. Create Runtime Endpoint
"""

import boto3
import json
import os
import subprocess
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv("/Users/haseeb/Code/iisc/bedrockAgentCore/.env")

# ─── Configuration ──────────────────────────────────────────────────────────
REGION = "ap-south-1"
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required locally to deploy.")
    
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]
RUNTIME_NAME = "simple_hello_agent"
ROLE_NAME = "BedrockAgentCoreExecutionRole"
ECR_REPO_NAME = "bedrock-agentcore/simple-agent"
IMAGE_TAG = "latest"

# ─── Clients ────────────────────────────────────────────────────────────────
iam = boto3.client("iam")
ecr = boto3.client("ecr", region_name=REGION)
agentcore = boto3.client("bedrock-agentcore-control", region_name=REGION)


def step(n, msg):
    print(f"\n{'='*60}")
    print(f"  Step {n}: {msg}")
    print(f"{'='*60}")


def run(cmd, check=True):
    """Run a shell command and print output."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")
    if result.returncode != 0 and check:
        print(f"  ❌ Command failed: {result.stderr.strip()}")
        raise RuntimeError(result.stderr)
    return result


# ─── Step 1: IAM Role ──────────────────────────────────────────────────────
def ensure_iam_role():
    step(1, "Ensure IAM execution role exists")

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "bedrock-agentcore.amazonaws.com"
                },
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {"aws:SourceAccount": ACCOUNT_ID}
                },
            }
        ],
    }

    try:
        role = iam.get_role(RoleName=ROLE_NAME)
        role_arn = role["Role"]["Arn"]
        print(f"  ✅ Role exists: {role_arn}")
    except iam.exceptions.NoSuchEntityException:
        print(f"  Creating role: {ROLE_NAME}")
        role = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Execution role for Bedrock AgentCore Runtime agents",
            Tags=[{"Key": "Project", "Value": "bedrockAgentCore"}],
        )
        role_arn = role["Role"]["Arn"]

        # Attach Bedrock model invocation permissions
        iam.attach_role_policy(
            RoleName=ROLE_NAME,
            PolicyArn="arn:aws:iam::aws:policy/AmazonBedrockFullAccess",
        )
        print(f"  ✅ Role created: {role_arn}")
        print("  ⏳ Waiting 10s for IAM propagation...")
        time.sleep(10)

    return role_arn


# ─── Step 2: Build Docker Image ────────────────────────────────────────────
def build_docker_image():
    step(2, "Build Docker image")

    image_name = f"{ECR_REPO_NAME}:{IMAGE_TAG}"
    # Use the directory of the script as the build context
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run(f"docker build --platform linux/arm64 -t {image_name} {script_dir}")
    print(f"  ✅ Image built: {image_name}")
    return image_name


# ─── Step 3: Push to ECR ───────────────────────────────────────────────────
def push_to_ecr(image_name):
    step(3, "Push image to ECR")

    ecr_uri = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com"
    full_uri = f"{ecr_uri}/{ECR_REPO_NAME}:{IMAGE_TAG}"

    # Create ECR repo if needed
    try:
        ecr.describe_repositories(repositoryNames=[ECR_REPO_NAME])
        print(f"  ✅ ECR repo exists: {ECR_REPO_NAME}")
    except ecr.exceptions.RepositoryNotFoundException:
        print(f"  Creating ECR repo: {ECR_REPO_NAME}")
        ecr.create_repository(
            repositoryName=ECR_REPO_NAME,
            imageScanningConfiguration={"scanOnPush": False},
        )
        print(f"  ✅ ECR repo created")

    # Login to ECR
    run(
        f"aws ecr get-login-password --region {REGION} | "
        f"docker login --username AWS --password-stdin {ecr_uri}"
    )

    # Tag and push
    run(f"docker tag {image_name} {full_uri}")
    run(f"docker push {full_uri}")

    print(f"  ✅ Pushed: {full_uri}")
    return full_uri


# ─── Step 4: Create Agent Runtime ──────────────────────────────────────────
def create_or_update_runtime(role_arn, container_uri):
    step(4, "Create AgentCore Runtime")

    # Check if runtime already exists
    try:
        runtimes = agentcore.list_agent_runtimes()
        for rt in runtimes.get("agentRuntimes", runtimes.get("agentRuntimeSummaries", [])):
            if rt.get("agentRuntimeName") == RUNTIME_NAME:
                runtime_id = rt["agentRuntimeId"]
                print(f"  ⚠️  Runtime already exists: {runtime_id}")
                print(f"  Updating with new container...")
                agentcore.update_agent_runtime(
                    agentRuntimeId=runtime_id,
                    roleArn=role_arn,
                    networkConfiguration={"networkMode": "PUBLIC"},
                    agentRuntimeArtifact={
                        "containerConfiguration": {
                            "containerUri": container_uri,
                        }
                    },
                    environmentVariables={
                        "AWS_DEFAULT_REGION": REGION,
                        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
                    },
                )
                print(f"  ✅ Runtime updated: {runtime_id}")
                wait_for_runtime(runtime_id)
                return runtime_id
    except Exception as e:
        print(f"  (Error during runtime check/update: {e})")
        raise

    # Create new runtime
    print(f"  Creating runtime: {RUNTIME_NAME}")
    response = agentcore.create_agent_runtime(
        agentRuntimeName=RUNTIME_NAME,
        description="Simple hello-world agent for testing AgentCore",
        roleArn=role_arn,
        networkConfiguration={"networkMode": "PUBLIC"},
        agentRuntimeArtifact={
            "containerConfiguration": {
                "containerUri": container_uri,
            }
        },
        environmentVariables={
            "AWS_DEFAULT_REGION": REGION,
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        },
        tags={"Project": "bedrockAgentCore"},
    )

    runtime_id = response["agentRuntimeId"]
    runtime_arn = response.get("agentRuntimeArn", "N/A")
    print(f"  ✅ Runtime created: {runtime_id}")
    print(f"     ARN: {runtime_arn}")

    wait_for_runtime(runtime_id)
    return runtime_id


def wait_for_runtime(runtime_id):
    """Wait for runtime to become ACTIVE."""
    print("  ⏳ Waiting for runtime to become ACTIVE...")
    for i in range(60):
        status = agentcore.get_agent_runtime(agentRuntimeId=runtime_id)
        state = status.get("status", "UNKNOWN")
        print(f"     Status: {state} (check #{i+1})", end="\r")
        if state in ("ACTIVE", "READY"):
            print(f"\n  ✅ Runtime is ACTIVE")
            return True
        elif state in ["FAILED", "DELETING"]:
            reason = status.get("failureReason", "unknown")
            print(f"\n  ❌ Runtime failed: {reason}")
            return False
        time.sleep(10)
    print(f"\n  ⚠️  Timed out waiting for runtime")
    return False


# ─── Step 5: Create Endpoint ───────────────────────────────────────────────
def create_or_get_endpoint(runtime_id):
    step(5, "Create Runtime Endpoint")

    endpoint_name = "default"

    # Check if endpoint exists
    try:
        endpoints = agentcore.list_agent_runtime_endpoints(
            agentRuntimeId=runtime_id
        )
        ep_list = endpoints.get("runtimeEndpoints", endpoints.get("agentRuntimeEndpointSummaries", []))
        print(f"  Found endpoints: {[ep.get('name') for ep in ep_list]}")
        for ep in ep_list:
            if ep.get("name").lower() == endpoint_name.lower():
                ep_id = ep.get("id", ep.get("agentRuntimeEndpointId", "N/A"))
                print(f"  ✅ Endpoint already exists: {ep_id} (status: {ep.get('status')})")
                return ep
    except Exception as e:
        print(f"  (Could not list endpoints: {e})")

    print(f"  Creating endpoint: {endpoint_name}")
    response = agentcore.create_agent_runtime_endpoint(
        agentRuntimeId=runtime_id,
        name=endpoint_name,
        description="Default endpoint for simple agent",
    )

    endpoint_id = response.get("agentRuntimeEndpointId", "N/A")
    print(f"  ✅ Endpoint created: {endpoint_id}")

    # Wait for endpoint
    print("  ⏳ Waiting for endpoint to become ACTIVE...")
    for i in range(60):
        try:
            ep_status = agentcore.get_agent_runtime_endpoint(
                agentRuntimeId=runtime_id,
                agentRuntimeEndpointId=endpoint_id,
            )
            state = ep_status.get("status", "UNKNOWN")
            print(f"     Status: {state} (check #{i+1})", end="\r")
            if state == "ACTIVE":
                print(f"\n  ✅ Endpoint is ACTIVE")
                return ep_status
            elif state in ["FAILED", "DELETING"]:
                reason = ep_status.get("failureReason", "unknown")
                print(f"\n  ❌ Endpoint failed: {reason}")
                return None
        except Exception:
            pass
        time.sleep(10)

    return response


# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"🚀 Deploying '{RUNTIME_NAME}' to AgentCore Runtime (container)")
    print(f"   Region:  {REGION}")
    print(f"   Account: {ACCOUNT_ID}")
    print(f"   Time:    {datetime.now().isoformat()}")

    role_arn = ensure_iam_role()
    image_name = build_docker_image()
    container_uri = push_to_ecr(image_name)
    runtime_id = create_or_update_runtime(role_arn, container_uri)

    if runtime_id:
        endpoint = create_or_get_endpoint(runtime_id)

        print(f"\n{'='*60}")
        print(f"  🎉 Deployment Complete!")
        print(f"{'='*60}")
        print(f"  Runtime ID:    {runtime_id}")
        print(f"  Container URI: {container_uri}")
        print(f"  Region:        {REGION}")
        print(f"\n  Next: Run invoke_simple.py to test the deployed agent")
    else:
        print("\n❌ Deployment failed. Check CloudWatch logs.")
