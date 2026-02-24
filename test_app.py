import os
from dotenv import load_dotenv

# load environment
load_dotenv("/Users/haseeb/Code/iisc/bedrockAC/.env")

# Must import from within the same directory context or ensure sys.path is correct
import sys
sys.path.insert(0, "/Users/haseeb/Code/iisc/bedrockAC")
sys.path.insert(0, "/Users/haseeb/Code/iisc/bedrockAC/workflows/react_agent_mcp")

from workflows.react_agent_mcp.app import handle

if __name__ == "__main__":
    print("Testing handle directly...")
    result = handle({"prompt": "Download the paper Attention Is All You Need and summarize its key contributions."})
    print("Result:", result)
