#!/bin/bash
# ============================================================================
# LaViRA Navigation System - Unitree G1 Humanoid
# ============================================================================
# Quick start script for running VLN navigation on G1.
#
# Prerequisites:
#   1. G1 robot powered on and connected via Ethernet
#   2. Orbbec Gemini 336L cameras connected and recognized
#   3. iPlanner server running (see run_iplanner_server.sh)
#   4. LLM endpoint configured via environment variables
#
# Usage:
#   ./scripts/run_navigation.sh "Go to the kitchen and find the red cup"
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default parameters
NETWORK_INTERFACE="${NETWORK_INTERFACE:-eth0}"
IPLANNER_URL="${IPLANNER_URL:-http://localhost:8888}"
TASK="${TASK:-vln}"
INSTRUCTION="${1:-Find the nearest exit}"

echo "============================================"
echo "  LaViRA Navigation - Unitree G1 Humanoid"
echo "============================================"
echo "Task:        $TASK"
echo "Instruction: $INSTRUCTION"
echo "Network:     $NETWORK_INTERFACE"
echo "iPlanner:    $IPLANNER_URL"
echo "============================================"

# Check LA model environment variables
if [ -z "$LA_API_KEY" ] && [ -z "$VA_API_KEY" ]; then
    echo "INFO: LA_API_KEY and VA_API_KEY not set."
    echo "      Local inference (llama-server) does not require API keys."
    echo "      For remote API endpoints set LA_API_KEY / VA_API_KEY."
fi

# Check iPlanner server
echo "Checking iPlanner server..."
if curl -s "$IPLANNER_URL/health" > /dev/null 2>&1; then
    echo "iPlanner server is running."
else
    echo "WARNING: iPlanner server not reachable at $IPLANNER_URL"
    echo "Start it with: ./scripts/run_iplanner_server.sh"
fi

# Run navigation
cd "$PROJECT_DIR"
python3 main.py \
    --task "$TASK" \
    --instruction "$INSTRUCTION" \
    --network_interface "$NETWORK_INTERFACE" \
    --iplanner_url "$IPLANNER_URL"
