#!/bin/bash
#SBATCH --job-name=FSReport
#SBATCH --output=/home/data/%u/data/MC_Hubbard_V6/sim_logs/filesystem_reports/filesystem_report_%j.out
#SBATCH --error=/home/data/%u/data/MC_Hubbard_V6/sim_logs/filesystem_reports/filesystem_report_%j.error
#SBATCH --time=01:00:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH -A dror-account
#SBATCH -p drorq

# ── Filesystem Size Report (SLURM) ──
# Usage:
#   sbatch submit_filesystem_report.sh                    # scan current dir
#   sbatch submit_filesystem_report.sh /path/to/scan      # scan specific dir
#   sbatch submit_filesystem_report.sh /path output.txt   # custom output file

SCAN_DIR="${1:-.}"
OUTFILE="${2:-${V6_DATA_ROOT:-/home/data/$USER/data/MC_Hubbard_V6}/sim_logs/filesystem_reports/filesystem_report_$(date +%Y%m%d_%H%M%S).txt}"

echo "=== Filesystem Report Job ==="
echo "Date     : $(date)"
echo "Node     : $(hostname)"
echo "Scan dir : $(realpath $SCAN_DIR)"
echo "Output   : $OUTFILE"
echo ""

julia systemsize_measure.jl "$SCAN_DIR" "$OUTFILE"

echo ""
echo "=== Job finished at $(date) ==="
