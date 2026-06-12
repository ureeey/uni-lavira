#!/bin/bash
# run_llama_server.sh — Launch a local llama.cpp server (llama-server) for Uni-LaViRA.
#
# Both the Language-Action (LA) and Vision-Action (VA) slots default to a single
# local Qwen3.5-27B-Q4_K_M GGUF served by llama.cpp's OpenAI-compatible
# `llama-server` at http://localhost:8000/v1. Because the model is multimodal,
# the VA image calls require the vision projector (`--mmproj`).
#
# Override any variable before running, e.g.:
#   MODEL_PATH=./models/Qwen3.5-27B-Q4_K_M.gguf MMPROJ_PATH=./models/mmproj.gguf bash run_llama_server.sh
#
# Environment variables
# ---------------------
#   MODEL_PATH   Path to the GGUF weights (default ./models/Qwen3.5-27B-Q4_K_M.gguf).
#   MMPROJ_PATH  Path to the multimodal projector GGUF (required for VA image calls).
#   MODEL_NAME   Served model name advertised on the OpenAI API; clients must use
#                this as LA_MODEL_NAME / VA_MODEL_NAME (default Qwen3.5-27B-Q4_K_M).
#   PORT         OpenAI-compatible API port (default 8000).
#   NGL          Number of layers to offload to the GPU (default 999 = all).
#   CTX          Context window size in tokens (default 8192).

MODEL_PATH="${MODEL_PATH:-./models/Qwen3.5-27B-Q4_K_M.gguf}"
MMPROJ_PATH="${MMPROJ_PATH:-}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B-Q4_K_M}"
PORT="${PORT:-8000}"
NGL="${NGL:-999}"
CTX="${CTX:-8192}"

echo "Starting llama.cpp server (llama-server)..."
echo "  Model      : ${MODEL_PATH}"
echo "  Served name: ${MODEL_NAME}  |  port: ${PORT}"

MMPROJ_ARGS=""
if [ -n "${MMPROJ_PATH}" ]; then
    MMPROJ_ARGS="--mmproj ${MMPROJ_PATH}"
fi

# llama-server exposes an OpenAI-compatible API at http://<host>:${PORT}/v1
llama-server \
    --model "${MODEL_PATH}" \
    ${MMPROJ_ARGS} \
    --alias "${MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --ctx-size "${CTX}" \
    --n-gpu-layers "${NGL}"
