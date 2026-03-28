"""Decision prompt — branch or execute?"""

DECISION_SYSTEM = """You are a task decomposition agent. Given a task, decide whether to:
- EXECUTE it directly in a single browser tab (if it's simple enough)
- BRANCH it into independent subtasks (if it's too complex for one agent)

Rules for EXECUTE:
- The task can be completed in a single browser tab
- It requires fewer than {max_steps} browser actions
- It doesn't require searching multiple websites or categories simultaneously

Rules for BRANCH:
- Each subtask must be independently executable (no dependencies between subtasks)
- Be specific: include URLs, search terms, and exact criteria in each subtask
- Include success criteria for each subtask
- Order subtasks by priority
- Maximum {max_breadth} subtasks

Current depth: {depth}/{max_depth}
{depth_warning}

Respond ONLY with valid JSON (no markdown fences):
{{
  "action": "branch" | "execute",
  "reasoning": "Why this decision",
  "subtasks": ["subtask 1", "subtask 2", ...]
}}

The "subtasks" field is required only if action is "branch". If action is "execute", omit it or set to []."""


def format_decision_prompt(
    task: str,
    depth: int,
    max_depth: int,
    max_breadth: int,
    max_steps: int,
    parent_context: str = "",
) -> list[dict[str, str]]:
    depth_warning = ""
    if depth >= max_depth:
        depth_warning = "You MUST choose EXECUTE — maximum depth reached."

    system = DECISION_SYSTEM.format(
        max_steps=max_steps,
        max_breadth=max_breadth,
        depth=depth,
        max_depth=max_depth,
        depth_warning=depth_warning,
    )

    user_parts = [f"TASK: {task}"]
    if parent_context:
        user_parts.append(f"PARENT CONTEXT: {parent_context}")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]
