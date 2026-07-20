# --- Per-step VLM decision prompt for rollout_v4 ---
# Single-call: model directly outputs decision + single best bbox.
# No JSON, no constraints, no hand-holding.

# PROMPT_V4 = """The current views: front, right, back, left.
# {came_from_fact}
# A — the {target} exists exactly in the views and within reach of arm
# B — the {target} exists exactly in the views and without reach of arm
# C — the {target} is out of the views exactly, but there is promising unexplored area 
# D — none of the above
# You are forbidden to select A or B unless you can confirm the {target} is definitely and unambiguously visible.
# Any hesitation, blur, partial view, lookalike objects, indirect hints all count as unconfirmed.
# Respond with exactly one line:
#   <A|B|C>,<front|right|back|left>,<x1>,<y1>,<x2>,<y2> or D
# 0-1000 per-mille"""

# PROMPT_V4 = """The current views: front, right, back, left.
# {came_from_fact}
# A — the {target} exists exactly in the views and within reach of arm
# B — the {target} exists exactly in the views and without reach of arm
# C — the {target} is out of the views exactly, but there is promising unexplored area 
# D — none of the above
# Respond with exactly one line:
#   <A|B|C>,<front|right|back|left>,<x1>,<y1>,<x2>,<y2> or D
# 0-1000 per-mille"""

# PROMPT_V4 = """A robot is performing an Object Navigation task. The target is: {target}.
# Below are the four directional views it currently observes: front , right , back , left .{came_from_fact}
# You are forbidden to select A or B unless you can confirm the {target} is definitely and unambiguously visible.
# Any hesitation, blur, partial view, lookalike objects, indirect hints all count as unconfirmed.
# Please select the most appropriate option from the following:

# A. The target is visible in at least one view AND is close enough to reach out and touch;
# B. The target is visible in at least one view, but is far away (not within arm's reach);
# C. The target is NOT visible in any view, but there are unexplored areas worth exploring (e.g., doorways, corridors, passages, open spaces that may lead to the target);
# D. None of the above (e.g., dead end, all views blocked, no explorable areas).

# Respond with exactly one line:
#   <A|B|C>,<front|right|back|left>,<x1>,<y1>,<x2>,<y2> or D
# 0-1000 per-mille"""

PROMPT_V4_PRE = """You are a robot performing an Object Navigation task.
Below are the four directional views it currently observes(front, right, back, left):"""

PROMPT_V4_HISTORY_HEADER = "Below are the previously explored areas(indexes start at 1):"

PROMPT_V4_POST = """Options:
A. There is a {target} in views within a distance of one meter;
B. There is a {target} in views;
C. There is no {target} in views, but there is at least one unexplored area worth exploring (e.g., doorways, corridors, passages, open spaces connected to the floor that may contain a {target});
D. None of the above (e.g., dead end, all views blocked, no explorable areas).

Rules:
You are forbidden to select A or B unless you can confirm a {target} is definitely and unambiguously visible.
Any hesitation, blur, partial view, lookalike objects, indirect hints all count as unconfirmed.
{came_from_fact}If you select C, avoid overly close meaningless or oversized indistinct areas such as blank walls.
If you select C, prefer the traversable areas covering the floor and ground surfaces without ceilings, sky and a closed door.
If you select C, check against the previously explored areas and mark as D if duplicated.

Please select the most appropriate option and respond with exactly one line:
  <A|B|C>,<front|right|back|left>,<x1>,<y1>,<x2>,<y2> or D,<ID of the image with the duplicated previously explored area>
0-1000 per-mille
"""


