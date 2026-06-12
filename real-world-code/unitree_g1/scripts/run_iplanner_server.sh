#!/bin/bash
# ============================================================================
# iPlanner Server Launcher for LaViRA G1
# ============================================================================
# Starts the iPlanner trajectory planning server.
# Must be running before starting navigation.
#
# Usage:
#   ./scripts/run_iplanner_server.sh [--port 8888] [--device cpu]
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IPLANNER_DIR="$PROJECT_DIR/iplanner"

PORT="${PORT:-8888}"
DEVICE="${DEVICE:-cpu}"

echo "============================================"
echo "  iPlanner Server - LaViRA G1"
echo "============================================"
echo "Port:       $PORT"
echo "Device:     $DEVICE"
echo "Config:     $IPLANNER_DIR/configs/iplanner.yaml"
echo "Checkpoint: $IPLANNER_DIR/checkpoints/iplanner.pth"
echo "============================================"

# Check checkpoint exists
if [ ! -f "$IPLANNER_DIR/checkpoints/iplanner.pth" ]; then
    echo "ERROR: iPlanner checkpoint not found!"
    echo "Expected: $IPLANNER_DIR/checkpoints/iplanner.pth"
    exit 1
fi

cd "$IPLANNER_DIR"
python3 iplanner_server.py \
    --port "$PORT" \
    --config ./configs/iplanner.yaml \
    --checkpoint ./checkpoints/iplanner.pth \
    --device "$DEVICE"
