"""Confidence evaluation prompt."""

EVALUATOR_SYSTEM = """Evaluate whether this result fully and correctly answers the given task.

Score from 0.0 to 1.0:
- 1.0: Complete, accurate, well-structured answer with all requested data
- 0.7-0.9: Mostly complete, minor gaps or uncertainties
- 0.4-0.6: Partial answer, significant gaps
- 0.1-0.3: Barely relevant or mostly wrong
- 0.0: Complete failure or empty result

Respond ONLY with valid JSON (no markdown fences):
{ "score": 0.0-1.0, "reasoning": "..." }"""


def format_evaluator_prompt(
    task: str,
    result: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": EVALUATOR_SYSTEM},
        {"role": "user", "content": f"TASK: {task}\n\nRESULT: {result}"},
    ]
