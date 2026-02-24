import sys
from app import handle
import json

prompt = "Get and sumarize the introduction and core contributions of the paper - Multi-scale competition in the Majorana-Kondo system"
print(f"\n[Prompt] {prompt}\n")
result = handle({"prompt": prompt})
print(f"\n[Response]\n{result.get('response', result.get('error'))}")
