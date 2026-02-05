import json
import re

def is_valid_json(text):
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def extract_json(text):
    """
    Extract JSON from messy LLM output
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def repair_json(llm_call, broken_json):
    """
    Ask LLM to repair invalid JSON
    """
    repair_prompt = f"""
You are a JSON repair assistant.
Fix the following JSON and return ONLY valid JSON.
No explanations.

Broken JSON:
{broken_json}
"""
    return llm_call(repair_prompt, temperature=0.0)
