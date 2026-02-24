
import boto3
import json

REGION = "ap-south-1"
client = boto3.client("bedrock-agentcore-control", region_name=REGION)

print(f"Checking runtimes in {REGION}...")
try:
    sts = boto3.client("sts")
    identity = sts.get_caller_identity()
    print(f"Caller Identity: {identity['Arn']}")
    print(f"Account: {identity['Account']}")

    runtimes = client.list_agent_runtimes()
    summaries = runtimes.get("agentRuntimes", runtimes.get("agentRuntimeSummaries", []))
    if not summaries:
        print("No runtimes found.")
    
    for rt in summaries:
        rt_id = rt["agentRuntimeId"]
        name = rt["agentRuntimeName"]
        status = rt["status"]
        print(f"Runtime: {name} ({rt_id}) - Status: {status}")
        
        # Get details
        details = client.get_agent_runtime(agentRuntimeId=rt_id)
        if status == "FAILED":
            print(f"  Failure Reason: {details.get('failureReason')}")
        
        # Get endpoints
        endpoints = client.list_agent_runtime_endpoints(agentRuntimeId=rt_id)
        ep_summaries = endpoints.get("agentEndpoints", endpoints.get("agentRuntimeEndpointSummaries", []))
        for ep in ep_summaries:
            ep_id = ep["agentRuntimeEndpointId"]
            ep_name = ep["name"]
            ep_status = ep["status"]
            print(f"  Endpoint: {ep_name} ({ep_id}) - Status: {ep_status}")
            if ep_status == "FAILED":
                ep_details = client.get_agent_runtime_endpoint(agentRuntimeId=rt_id, agentRuntimeEndpointId=ep_id)
                print(f"    Failure Reason: {ep_details.get('failureReason')}")

except Exception as e:
    print(f"Error: {e}")
