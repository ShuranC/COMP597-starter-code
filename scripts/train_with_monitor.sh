#!/bin/bash
# Runs GPU/CPU monitoring in the background within the same srun job as training.
# Usage: pass all launch.sh arguments after --
# Example: ./scripts/bash_srun.sh "./scripts/train_with_monitor.sh bs32 -- --model regnet ..."

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

LABEL=${1}; shift
shift  # consume the '--'

OUTPUT_DIR=/mnt/teaching/slurm/scui4/results
GPU_OUT="${OUTPUT_DIR}/${LABEL}_gpu_timeline.csv"
CPU_OUT="${OUTPUT_DIR}/${LABEL}_cpu_timeline.csv"

# Start GPU monitoring
echo "timestamp,gpu_util_pct,mem_used_mib,power_w" > "${GPU_OUT}"
nvidia-smi dmon -s upw -d 1 -o T | awk '
/^#/ { next }
/^[0-9]/ { printf "%s %s,%s,%s,%s\n", $1, $2, $3, $5, $7 }
' >> "${GPU_OUT}" &
NVID_PID=$!

# Start CPU monitoring
echo "timestamp,cpu_util_pct" > "${CPU_OUT}"
while true; do
    TS=$(date +"%Y-%m-%dT%H:%M:%S")
    CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    echo "${TS},${CPU}" >> "${CPU_OUT}"
    sleep 1
done &
CPU_PID=$!

echo "Monitoring started (GPU PID=${NVID_PID}, CPU PID=${CPU_PID})"

# Run training
cd ${REPO_DIR}
python3 launch.py "$@"
EXIT_CODE=$?

# Stop monitoring
kill ${NVID_PID} ${CPU_PID} 2>/dev/null
echo "Monitoring stopped."

exit ${EXIT_CODE}
