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

PROMPT_V4 = """A robot is performing an Object Navigation task. The target is: {target}.
Below are the four directional views it currently observes: front , right , back , left .{came_from_fact}
You are forbidden to select A or B unless you can confirm the {target} is definitely and unambiguously visible.
Any hesitation, blur, partial view, lookalike objects, indirect hints all count as unconfirmed.
Avoid selecting overly close meaningless areas and oversized indistinct areas.
Please select the most appropriate option from the following:

A. The target is visible in at least one view AND is close enough to reach out and touch;
B. The target is visible in at least one view, but is far away (not within arm's reach);
C. The target is NOT visible in any view, but there are unexplored areas worth exploring (e.g., doorways, corridors, passages, open spaces that may lead to the target);
D. None of the above (e.g., dead end, all views blocked, no explorable areas).

Respond with exactly one line:
  <A|B|C>,<front|right|back|left>,<x1>,<y1>,<x2>,<y2> or D
0-1000 per-mille"""

PROMPT_V4_HISTORY_HEADER = "Previously explored:"
