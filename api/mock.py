import re

def call(prompt, model, api_key):
    blocks = re.split(r"\n\d+:\n", prompt)[1:]
    return "\n".join(f"{i+1}:\n{block.strip()}" for i, block in enumerate(blocks))
