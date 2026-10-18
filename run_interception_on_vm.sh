#!/usr/bin/env bash
set -e

# Source ROS 2 Jazzy environment
source /opt/ros/jazzy/setup.bash 2>/dev/null || true

echo "================================================================================"
echo "[AetherLab Cloud Sim] Starting Counter-UAV Interception Benchmark Suite"
echo "Host: $(uname -a)"
echo "Timestamp: $(date -u)"
echo "================================================================================"

echo "[1/4] Cleaning previous simulation processes..."
killall -9 gz-sim-server ruby gz ros2 parameter_bridge 2>/dev/null || true
sleep 1

echo "[2/4] Verifying Python3 and NumPy environment..."
python3 -c "import numpy; print('NumPy version:', numpy.__version__)"

echo "[3/4] Executing 3D Pursuit-Evasion Scale-Dilation TPN Benchmark..."
python3 /tmp/interception_sim_node.py --benchmark --output /tmp/sim_metrics.csv

echo "[4/4] Validating output telemetry..."
if [ -f "/tmp/sim_metrics.csv" ]; then
    echo "Telemetry CSV generated successfully. File size:"
    ls -lh /tmp/sim_metrics.csv
    echo "Total telemetry rows:"
    wc -l /tmp/sim_metrics.csv
    head -n 5 /tmp/sim_metrics.csv
else
    echo "Error: /tmp/sim_metrics.csv was not generated!"
    exit 1
fi

echo "================================================================================"
echo "[AetherLab Cloud Sim] Run complete. Ready for telemetry download."
echo "================================================================================"
