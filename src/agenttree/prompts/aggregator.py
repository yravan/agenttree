"""Aggregation prompt — combine child results."""

AGGREGATION_SYSTEM = """You are a result aggregator. Your task was decomposed into subtasks that have now completed.

Instructions:
1. Combine all successful results into a single comprehensive answer
2. Deduplicate entries (same item from different sources)
3. Flag any gaps where subtasks failed or returned low-confidence results
4. If critical gaps remain, list additional subtasks that would fill them

Respond ONLY with valid JSON (no markdown fences):
{
  "final_result": { ... },
  "confidence": 0.0-1.0,
  "completeness": 0.0-1.0,
  "gaps": ["description of gap 1", ...],
  "needs_more_work": true/false,
  "additional_subtasks": ["subtask 1", ...]
}"""


def format_aggregation_prompt(
    task: str,
    child_results: list[dict],
) -> list[dict[str, str]]:
    results_text = []
    for i, child in enumerate(child_results, 1):
        entry = f"""--- Subtask {i} ---
Task: {child.get('task', 'N/A')}
Status: {child.get('status', 'N/A')}
Confidence: {child.get('confidence', 'N/A')}
Result: {child.get('result', 'N/A')}"""
        if child.get("error"):
            entry += f"\nError: {child['error']}"
        results_text.append(entry)

    user_content = f"""ORIGINAL TASK: {task}

SUBTASK RESULTS:
{chr(10).join(results_text)}"""

    return [
        {"role": "system", "content": AGGREGATION_SYSTEM},
        {"role": "user", "content": user_content},
    ]
