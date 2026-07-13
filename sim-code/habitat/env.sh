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

# --- Logging verbosity ---
# LAVIRA_LOG_PROMPT:  0=full (default), 1=skip prompt templates, 2=mute all prompt/output
# LAVIRA_LOG_VERBOSE:  0=full (default), 1=quiet (no ChatCompletion dumps etc.)
# LAVIRA_LOG_NETWORK:  0=off (default), 1=log proxy state & per-request network diagnostics
# HABITAT_SIM_LOG:     silence C++ habitat-sim INFO logs (set to "warning" for quiet)
# Uncomment to enable:
export LAVIRA_LOG_PROMPT=1
export LAVIRA_LOG_VERBOSE=1
export GLOG_minloglevel=1
export LAVIRA_LOG_NETWORK=1
# Alternative if glog doesn't work: export HABITAT_SIM_LOG=warning
