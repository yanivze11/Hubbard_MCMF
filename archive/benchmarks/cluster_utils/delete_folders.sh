#!/bin/bash

#SBATCH --job-name=delete_large_folders
#SBATCH --output=/home/data/%u/data/MC_Hubbard_V6/sim_logs/folder_deleter/delete_log_%j.out
#SBATCH --error=/home/data/%u/data/MC_Hubbard_V6/sim_logs/folder_deleter/delete_log_%j.error
#SBATCH --account=dror-account
#SBATCH --partition=drorq           
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1G                    

# --- CONFIGURATION ---
TARGET_DIR="${V6_DATA_ROOT:-/home/data/$USER/data/MC_Hubbard_V6}/data/Mag_L10__19-01-2026"
# EXCLUDE1="num_control"
# EXCLUDE2="results"
EXCLUDE1=""
EXCLUDE2=""
# ---------------------

echo "Starting deletion of contents in $TARGET_DIR (excluding '$EXCLUDE1' and '$EXCLUDE2') at $(date)"

if [ -d "$TARGET_DIR" ]; then
    # Build find commands based on which exclusions are set
    if [ -n "$EXCLUDE1" ] && [ -n "$EXCLUDE2" ]; then
        # Both exclusions active
        find "$TARGET_DIR" -mindepth 1 -not -type d \
            -not -path "$TARGET_DIR/$EXCLUDE1/*" \
            -not -path "*/$EXCLUDE2/*" \
            -delete

        find "$TARGET_DIR" -mindepth 1 -depth -type d -empty \
            -not -path "$TARGET_DIR/$EXCLUDE1" \
            -not -path "$TARGET_DIR/$EXCLUDE1/*" \
            -not -name "$EXCLUDE2" \
            -delete

    elif [ -n "$EXCLUDE1" ]; then
        # Only EXCLUDE1 active
        find "$TARGET_DIR" -mindepth 1 -not -type d \
            -not -path "$TARGET_DIR/$EXCLUDE1/*" \
            -delete

        find "$TARGET_DIR" -mindepth 1 -depth -type d -empty \
            -not -path "$TARGET_DIR/$EXCLUDE1" \
            -not -path "$TARGET_DIR/$EXCLUDE1/*" \
            -delete

    elif [ -n "$EXCLUDE2" ]; then
        # Only EXCLUDE2 active
        find "$TARGET_DIR" -mindepth 1 -not -type d \
            -not -path "*/$EXCLUDE2/*" \
            -delete

        find "$TARGET_DIR" -mindepth 1 -depth -type d -empty \
            -not -name "$EXCLUDE2" \
            -delete

    else
        # No exclusions — delete everything
        rm -rf "$TARGET_DIR"/*
    fi

    echo "Deletion complete at $(date)"
else
    echo "Error: Directory $TARGET_DIR does not exist."
    exit 1
fi