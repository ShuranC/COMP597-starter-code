#!/bin/bash
# Usage: monitor_gpu.sh <output_dir> <label>
# Samples GPU and CPU utilization every 1 second until killed.
# Run this in the background alongside training, then kill it when done.

OUTPUT_DIR=${1:-/mnt/teaching/slurm/scui4/results}
LABEL=${2:-monitor}

GPU_OUT="${OUTPUT_DIR}/${LABEL}_gpu_timeline.csv"
CPU_OUT="${OUTPUT_DIR}/${LABEL}_cpu_timeline.csv"

# GPU: sm utilization (%), memory used (MiB), power (W), timestamp
echo "timestamp,gpu_util_pct,mem_used_mib,power_w" > "${GPU_OUT}"
nvidia-smi dmon -s upw -d 1 -o T | awk '
/^#/ { next }
/^[0-9]/ {
    # columns: date time idx sm_util mem_util enc dec power
    printf "%s %s,%s,%s,%s\n", $1, $2, $3, $5, $7
}
' >> "${GPU_OUT}" &
NVID_PID=$!

# CPU: overall utilization % sampled every 1 second
echo "timestamp,cpu_util_pct" > "${CPU_OUT}"
while true; do
    TS=$(date +"%Y-%m-%dT%H:%M:%S")
    CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    echo "${TS},${CPU}" >> "${CPU_OUT}"
    sleep 1
done &
CPU_PID=$!

echo "Monitoring started. GPU PID=${NVID_PID}, CPU PID=${CPU_PID}"
echo "Kill with: kill ${NVID_PID} ${CPU_PID}"
echo "${NVID_PID} ${CPU_PID}" > "${OUTPUT_DIR}/${LABEL}_monitor_pids.txt"

# Wait for both (they run until killed)
wait
