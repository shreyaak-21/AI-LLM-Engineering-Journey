# prompt_builder.py
def build_prompt(system_prompt, user_prompt):
    return f"""
[System Instruction]
{system_prompt}

[User Message]
{user_prompt}
"""
