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
