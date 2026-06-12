#!/bin/bash
# Quick Start Script for Indoor Navigation System

echo "=========================================="
echo "Indoor Navigation System - Quick Start"
echo "=========================================="
echo ""

# Check environment variables
echo "[1/5] Checking environment variables..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ ERROR: OPENAI_API_KEY not set (for Qwen2.5-VL-32B)"
    echo "   Please run: export OPENAI_API_KEY='your-key'"
    exit 1
fi

if [ -z "$OPENAI_API_KEY_SECONDARY" ]; then
    echo "❌ ERROR: OPENAI_API_KEY_SECONDARY not set (for Gemini-2.5-pro)"
    echo "   Please run: export OPENAI_API_KEY_SECONDARY='your-key'"
    exit 1
fi

echo "✓ Environment variables configured"
echo ""

# Check NavDP server
echo "[2/5] Checking NavDP server..."
if curl -s http://localhost:8888/health > /dev/null 2>&1; then
    echo "✓ NavDP server is running"
else
    echo "❌ ERROR: NavDP server not running"
    echo "   Please start it in another terminal:"
    echo "   cd <workspace>/src/NavDP/baselines/navdp"
    echo "   python navdp_server.py --port 8888 --checkpoint path/to/checkpoint.ckpt"
    exit 1
fi
echo ""

# Check ROS master
echo "[3/5] Checking ROS master..."
if rostopic list > /dev/null 2>&1; then
    echo "✓ ROS master is running"
else
    echo "❌ ERROR: ROS master not running"
    echo "   Please run: roscore"
    exit 1
fi
echo ""

# Check model_set_node
echo "[4/5] Checking model_set_node..."
if rostopic list | grep -q "/unilavira/nav_state"; then
    echo "✓ model_set_node is running"
else
    echo "⚠ WARNING: model_set_node may not be running"
    echo "   If not started, please run in another terminal:"
    echo "   roslaunch vln_node model_set.launch"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# Launch indoor navigation
echo "[5/5] Launching indoor navigation..."
echo ""
echo "=========================================="
echo "Starting Indoor Navigation System"
echo "=========================================="
echo ""

roslaunch vln_node indoor_eval.launch
