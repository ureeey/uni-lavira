# --- Per-step VLM decision prompts for rollout_v2 ---
# These are used by VLMReasoningAgentV2 in a multi-turn conversation chain.
# Each prompt appends to the previous messages, forming a dialogue with the VLM.

# PROMPT_IS_TARGET_VISIBLE = """Is {target} present in the photo? Answer yes only if you are completely certain, otherwise answer no. Answer only yes or no."""

PROMPT_IS_TARGET_VISIBLE = """
You are forbidden to answer yes unless you can confirm {target} is definitely and unambiguously visible in the photo.
Any hesitation, blur, partial view, lookalike objects, indirect hints all count as unconfirmed.
Unconfirmed → reply no.
Confirmed beyond any reasonable doubt → reply yes.
No explanations, only output yes or no.
"""

PROMPT_IS_TARGET_NEAR = """Is the photographer close enough to {target} to reach out and touch it? Answer only yes or no."""

PROMPT_TARGET_BBOX = """Mark the only one area in the photo. Respond using only the following format.
Response format (JSON):
{{"bbox_2d": [x1, y1, x2, y2]}}"""

PROMPT_IS_TARGET_POSSIBLE = """Is there at least one passable, explorable area in the current photo? Answer only yes or no."""

# PROMPT_IS_TARGET_POSSIBLE = """Is there at least one passable, explorable area in the current photo, where reaching it might lead to finding {target}? Answer only yes or no."""

# PROMPT_IS_TARGET_POSSIBLE = """Does the photo contain any open door, passage, corridor or accessible traversable space that can be entered or walked into, which could lead you to find {target}? Answer only yes or no."""

PROMPT_POSSIBLE_BBOX = """Mark the only one area with the highest probability of finding {target} in the photo. Respond using only the following format.
Response format (JSON):
{{"bbox_2d": [x1, y1, x2, y2]}}"""

PROMPT_IS_REPEAT_CURRENT = """This is the current photo with marked area:"""

PROMPT_IS_REPEAT_HISTORY = """Previously seen marked photos. Please consider both the entire photo and the marked area comprehensively. Do any of these marked photos show the same situation as the current photo? Answer only yes or no."""
