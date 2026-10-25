#!/bin/bash

#SBATCH --job-name=delete_omega_folders
#SBATCH --output=/home/data/%u/data/MC_Hubbard_V6/sim_logs/folder_deleter/delete_omega_%j.out
#SBATCH --error=/home/data/%u/data/MC_Hubbard_V6/sim_logs/folder_deleter/delete_omega_%j.error
#SBATCH --account=dror-account
#SBATCH --partition=drorq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1G

# --- CONFIGURATION ---
MAIN_DIR="${V6_DATA_ROOT:-/home/data/$USER/data/MC_Hubbard_V6}/data/FullSys__11-02-2026"
DRY_RUN=false   # Set to false to actually delete
# ---------------------

SUB_PATH="results/statistics_report/fixed_delta_observables"

echo "=== Omega folder cleanup ==="
echo "Main directory: $MAIN_DIR"
echo "Dry run: $DRY_RUN"
echo "Started at $(date)"
echo ""

if [ ! -d "$MAIN_DIR" ]; then
    echo "Error: Directory $MAIN_DIR does not exist."
    exit 1
fi

deleted_count=0
kept_count=0

# Iterate over all delta_* folders matching the structure
for delta_dir in "$MAIN_DIR"/*/"$SUB_PATH"/*/delta_*; do
    [ -d "$delta_dir" ] || continue

    # Collect omega_* subdirectories
    omega_dirs=()
    for d in "$delta_dir"/omega_*; do
        [ -d "$d" ] && omega_dirs+=("$d")
    done

    # Skip if 0 or 1 omega folders (nothing to delete)
    if [ ${#omega_dirs[@]} -le 1 ]; then
        continue
    fi

    # Sort omega dirs by numeric value (extract number after "omega_")
    # Find the lowest omega value
    lowest_val=""
    lowest_dir=""
    for d in "${omega_dirs[@]}"; do
        basename_d=$(basename "$d")
        val="${basename_d#omega_}"
        if [ -z "$lowest_val" ] || awk "BEGIN{exit !($val < $lowest_val)}"; then
            lowest_val="$val"
            lowest_dir="$d"
        fi
    done

    # Delete all omega dirs except the lowest
    for d in "${omega_dirs[@]}"; do
        if [ "$d" != "$lowest_dir" ]; then
            if [ "$DRY_RUN" = true ]; then
                echo "[DRY RUN] Would delete: $d"
            else
                echo "Deleting: $d"
                rm -rf "$d"
            fi
            ((deleted_count++))
        else
            if [ "$DRY_RUN" = true ]; then
                echo "[KEEP]    $d  (lowest omega=$lowest_val)"
            fi
            ((kept_count++))
        fi
    done
done

echo ""
echo "=== Summary ==="
if [ "$DRY_RUN" = true ]; then
    echo "Would delete $deleted_count omega folder(s), keeping $kept_count."
    echo "Set DRY_RUN=false to actually delete."
else
    echo "Deleted $deleted_count omega folder(s), kept $kept_count."
fi
echo "Finished at $(date)"
