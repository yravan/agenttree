"""Executor prompt — browser agent ReAct loop."""

EXECUTOR_SYSTEM = """You are a browser automation agent. Complete your assigned task using the available tools.

You have access to a Chrome browser tab. Use the tools to navigate, click, type, scroll, and extract data.

Important guidelines:
- Be methodical: think about what you see and what you need to do next
- Call "done" when finished, with your extracted data as the result
- If you encounter a login wall or CAPTCHA, call done(result="BLOCKED", success=false)
- If results seem empty when they shouldn't be, you may be soft-blocked — try a different approach
- Extract structured data whenever possible (JSON objects/arrays)
- Include real URLs in your results when available"""


def format_executor_prompt(
    task: str,
    step: int,
    max_steps: int,
    memory: str,
    recent_history: str,
    dom: str,
) -> list[dict[str, str]]:
    user_content = f"""TASK: {task}
STEP: {step}/{max_steps}

Memory (summary of earlier steps):
{memory or "(none)"}

Recent actions:
{recent_history or "(none)"}

Current page:
{dom or "(no page loaded)"}

Think about what you see and what you need to do next. Then call a tool.
Call "done" when finished, with your extracted data as the result."""

    return [
        {"role": "system", "content": EXECUTOR_SYSTEM},
        {"role": "user", "content": user_content},
    ]
