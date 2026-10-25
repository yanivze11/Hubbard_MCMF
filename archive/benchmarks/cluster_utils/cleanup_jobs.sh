#!/bin/bash
#SBATCH --job-name=cleanup_monitor
#SBATCH --output=/home/data/%u/data/MC_Hubbard_V6/sim_logs/folder_deleter/job_cleanup_%j.out
#SBATCH --error=/home/data/%u/data/MC_Hubbard_V6/sim_logs/folder_deleter/job_cleanup_%j.error
#SBATCH --time=00:05:00
#SBATCH --mem=100M
#SBATCH --account=dror-account
#SBATCH --partition=drorq   

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_DRY_RUN=true   # Default: only print what would happen.
DEFAULT_MAX_HOURS=72   # Default threshold in hours.
# ==========================================

usage() {
    echo "Usage: $0 [MAX_HOURS] [DRY_RUN]"
    echo "  MAX_HOURS: positive integer (default: $DEFAULT_MAX_HOURS)"
    echo "  DRY_RUN: true|false (default: $DEFAULT_DRY_RUN)"
}

# Positional inputs: [MAX_HOURS] [DRY_RUN]
MAX_HOURS="${1:-$DEFAULT_MAX_HOURS}"
DRY_RUN="${2:-$DEFAULT_DRY_RUN}"

if ! [[ "$MAX_HOURS" =~ ^[0-9]+$ ]] || [ "$MAX_HOURS" -le 0 ]; then
    echo "Error: MAX_HOURS must be a positive integer. Received: '$MAX_HOURS'"
    usage
    exit 1
fi

if [ "$DRY_RUN" != "true" ] && [ "$DRY_RUN" != "false" ]; then
    echo "Error: DRY_RUN must be 'true' or 'false'. Received: '$DRY_RUN'"
    usage
    exit 1
fi

MAX_SECONDS=$((MAX_HOURS * 3600))
USER_ID=$(whoami)

echo "Starting job cleanup script..."
echo "Dry Run Mode: $DRY_RUN"
echo "Time Threshold: $MAX_HOURS hours ($MAX_SECONDS seconds)"
echo "------------------------------------------------------"

# Get running jobs for the current user, outputting Job ID (%i) and Time Used (%M)
squeue -u "$USER_ID" -t RUNNING -h -o "%i %M" | while read jobid timeused; do
    
    # Slurm time format can be DD-HH:MM:SS, HH:MM:SS, or MM:SS
    # This awk command parses the timeused string and converts it to total seconds
    seconds=$(echo "$timeused" | awk -F'-|:' '{
        if (NF==4) print ($1*86400) + ($2*3600) + ($3*60) + $4
        else if (NF==3) print ($1*3600) + ($2*60) + $3
        else if (NF==2) print ($1*60) + $2
        else print 0
    }')

    # Compare running time against our threshold
    if [ "$seconds" -gt "$MAX_SECONDS" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] Would cancel job $jobid (Running for $timeused)"
        else
            echo "[ACTION] Canceling job $jobid (Running for $timeused)"
            scancel "$jobid"
        fi
    fi
done

echo "------------------------------------------------------"
echo "Cleanup script finished."