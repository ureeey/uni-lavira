# System & runtime config — safe to commit to git (no secrets).
# Source this together with .env.local:
#   source .env.local && source env.sh
#
# or simply:  source env.sh  (if API keys are already in your environment)

# --- HuggingFace offline mode ---
# Prevents transformers from trying to download model files from huggingface.co.
# The bert-base-uncased tokenizer & weights must be cached locally in
#   ~/.cache/huggingface/hub/models--bert-base-uncased/
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- NVIDIA PRIME render offload ---
# REQUIRED on hybrid-graphics (Intel + NVIDIA) laptops.  Without these,
# habitat-sim's OpenGL rendering runs on the Intel GPU and produces black
# frames because the scene textures are in NVIDIA VRAM.
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# --- Logging ---
# Master preset — sets sensible defaults for all categories below:
#   LAVIRA_LOG=quiet    results only (compact progress bars, minimal output)
#   LAVIRA_LOG=normal   prompts + decisions (suitable for daily eval)
#   LAVIRA_LOG=debug    everything including network, body, FMM, full request history
export LAVIRA_LOG=normal
export LAVIRA_LOG_REQ=0
export LAVIRA_LOG_RESP=0
export LAVIRA_LOG_PLAN=1
export LAVIRA_LOG_FMM=0
export LAVIRA_LOG_ACT=0
export LAVIRA_LOG_BODY=0
export LAVIRA_LOG_NETWORK=0
# Per-category overrides (take precedence over the preset):
#
# Evaluator layer (ZS_Evaluator_mp.py):
#   LAVIRA_LOG_PLAN=1   branch decisions & NAV steps         [0/1]
#   LAVIRA_LOG_FMM=1    FMM planner output                   [0/1]
#   LAVIRA_LOG_ACT=1    action execution                     [0/1]
#
# Agent layer (agent.py / agent_v2.py / agent_v3.py):
#   LAVIRA_LOG_REQ=1    API request content                  [0=off, 1=incremental, 2=full]
#   LAVIRA_LOG_RESP=1   prompt & response content            [0/1]
#
# API layer (api_openai.py / api_dashscope.py):
#   LAVIRA_LOG_BODY=1     HTTP body (image sizes, tokens)    [0/1]
#   LAVIRA_LOG_NETWORK=1  HTTP metadata (latency, status)    [0/1]
#
# Examples of selective overrides:
#   export LAVIRA_LOG=normal
#   export LAVIRA_LOG_FMM=1       # also show FMM output in normal mode
#   export LAVIRA_LOG_REQ=2       # full request history in normal mode

# Silence C++ habitat-sim INFO logs:
export GLOG_minloglevel=1
# Alternative if glog doesn't work: export HABITAT_SIM_LOG=warning
