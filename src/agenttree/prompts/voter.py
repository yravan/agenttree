"""Voter prompt — pick best result from N candidates."""

VOTER_SYSTEM = """You have multiple results for the same task. Pick the best one.

Evaluate each result on:
- Completeness: Does it fully answer the task?
- Accuracy: Is the information correct and verifiable?
- Structure: Is it well-organized and actionable?

Respond ONLY with valid JSON (no markdown fences):
{
  "best_index": 0,
  "confidence": 0.0-1.0,
  "reasoning": "Why this result is best"
}

best_index is 0-indexed."""


def format_voter_prompt(
    task: str,
    results: list,
) -> list[dict[str, str]]:
    results_text = []
    for i, result in enumerate(results):
        results_text.append(f"--- RESULT {i} ---\n{result}\n")

    user_content = f"""TASK: {task}

{chr(10).join(results_text)}"""

    return [
        {"role": "system", "content": VOTER_SYSTEM},
        {"role": "user", "content": user_content},
    ]
