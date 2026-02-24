import sys
from app import handle
import json

prompt = "Download the paper 'Attention Is All You Need' and summarize its key contributions."
print(f"\n[Prompt] {prompt}\n")
result = handle({"prompt": prompt})
print(f"\n[Response]\n{result.get('response', result.get('error'))}")
