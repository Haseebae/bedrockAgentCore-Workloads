import json
import requests
import uuid
import sys

def main():
    # URL of the deployed BedrockAgentCore app. Adjust the url/port if needed.
    url = "http://localhost:8080/"
    if len(sys.argv) > 1:
        url = sys.argv[1]

    # The plan structure the user wants to execute
    plan_dict = {
        "tools_to_use": [
            {
                "tool_name": "download_article",
                "purpose": "To download the PDF of the paper Multi-scale competition in the Majorana-Kondo system"
            },
            {
                "tool_name": "document_retriever",
                "purpose": "To retrieve and summarize the introduction and core contributions from the downloaded paper."
            }
        ],
        "reasoning": "First, I need to download the paper to access its content, then I can summarize the relevant sections."
    }

    # Format the payload based on what app.handle() expects
    payload = {
        "plan": json.dumps(plan_dict, indent=2),
        "prompt": "Please download the paper 'Multi-scale competition in the Majorana-Kondo system' and summarize the introduction and core contributions.",
        "session_id": "test_session_001",
        "trace_id": uuid.uuid4().hex,
        "state_id": uuid.uuid4().hex,
        "actor_id": "test_actor_001",
        "memory_config": "empty",
        "agent_state": {
            "iteration_count": 0,
            "step_count": 0
        }
    }

    headers = {
        "Content-Type": "application/json"
    }

    print(f"Sending POST request to {url}")
    print("Payload:")
    print(json.dumps(payload, indent=2))
    print("-" * 50)

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        print("\nResponse Status:", response.status_code)
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"\nError hitting endpoint: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print("Response JSON:", json.dumps(e.response.json(), indent=2))
            except json.JSONDecodeError:
                print("Response text:", e.response.text)

if __name__ == "__main__":
    main()
