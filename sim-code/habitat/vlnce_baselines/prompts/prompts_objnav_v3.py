# --- Per-step VLM decision prompts for rollout_v3 ---
# These are used by VLMReasoningAgentV3 in single-call style.
# Each call is standalone — all images are included in one request.

PROMPT_JUDGE = """A robot is performing an Object Navigation task. The target is: {target}.
Below are the four directional views it currently observes: Front (0), Right (1), Back (2), Left (3).{came_from_hint}
You are forbidden to select A unless you can confirm {target} is definitely and unambiguously visible.
Any hesitation, blur, partial view, lookalike objects, indirect hints all count as unconfirmed.
Please select the most appropriate option from the following:

A. The target is visible in at least one view AND is close enough to reach out and touch;
B. The target is visible in at least one view, but is far away (not within arm's reach);
C. The target is NOT visible in any view, but there are areas worth exploring (e.g., doorways, corridors, passages, open spaces that may lead to the target);
D. None of the above (e.g., dead end, all views blocked, no explorable areas).

You must first answer with exactly one letter: A, B, C, or D.

If you answer A or B or C, you must also provide a structured JSON object in the following format:
```json
{{
  "plan": "A",
  "regions": [
    {{
      "frame_idx": 0,
      "regions": [
        {{"idx": 0, "bbox": [x1, y1, x2, y2]}}
      ]
    }}
  ]
}}
```

CRITICAL — you MUST use exactly these JSON key names, no aliases:
- `frame_idx` (integer, required)
- `idx` (integer, required — NOT "label" or any other name)
- `bbox` (array of 4 integers in per-mille 0-1000, required — NOT "bbox_2d" or any other name)
- `plan` (string "A"/"B"/"C"/"D", required)

Additional rules:
- `frame_idx` is the view index: 0=Front, 1=Right, 2=Back, 3=Left.
- `idx` is the region index within that frame (0-based).
- `bbox` uses per-mille coordinates (0-1000), where [0, 0, 1000, 1000] is the entire image.
- When answering A (STOP) or B (APPROACH): provide EXACTLY ONE frame with EXACTLY ONE region — the one best showing the target.
- When answering C (EXPLORE): provide AT LEAST ONE frame, each with AT LEAST ONE region, covering all promising exploration areas across all views.
- When answering D (OTHER): set `regions` to an empty list `[]` and provide a brief `reason` field instead.

Respond with the letter first on its own line, then the JSON on subsequent lines.
Do NOT include any other text outside the letter and the JSON block."""

PROMPT_SELECT_ONE = """A robot is performing an Object Navigation task. The target is: {target}.

Below are the current candidate exploration areas with their frame_idx, bbox_idx, and marked bounding boxes.

Select the single most promising candidate that has NOT been explored before. Prefer areas leading to new, unseen regions. Avoid areas that look similar to any previously explored ones (shown in a separate message, if any).

Respond with exactly this JSON format:
```json
{{"frame_idx": <int>, "bbox_idx": <int>}}
```
If ALL candidates have already been explored, use null values:
```json
{{"frame_idx": null, "bbox_idx": null}}
```
Do NOT include any text outside the JSON block."""

PROMPT_SELECT_ONE_CANDIDATES_HEADER = "Candidate exploration areas (with hierarchical list for reference):"
PROMPT_SELECT_ONE_HISTORY_HEADER = "Previously explored areas — avoid these (already visited or rejected):"
