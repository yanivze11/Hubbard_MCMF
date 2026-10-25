#!/usr/bin/env bash
# Orchestrate the Gaussian-model build: one isolated Julia process per temperature
# (crash-safe; each T saved to its own file), then merge + validate.
# Run with -O1 + single-thread BLAS to avoid the transient LLVM bad_alloc crash.
set -u
cd "$(dirname "$0")/../.."   # the repository
RES="gaussian_model_results"
mkdir -p "$RES"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# T  delta(=measured Delta_MC_abs, full precision)
PAIRS=( "0.5 2.01897173782932" "1.0 2.476387218004027" "2.0 3.084972665857382" "4.0 3.992057196844593" )

for p in "${PAIRS[@]}"; do
  set -- $p; T=$1; D=$2
  OUT="$RES/entry_T${T}.jld2"
  if [ -f "$OUT" ]; then echo "=== T=$T already done ($OUT) — skip ==="; continue; fi
  echo "=== [$(date +%H:%M:%S)] running T=$T (Δ_p=$D) ==="
  julia -O1 gaussian_model_run.jl one-T "$T" "$D" "$OUT"
  echo "=== T=$T exit=$? ==="
done

echo "=== [$(date +%H:%M:%S)] merge + validate ==="
julia -O1 gaussian_model_run.jl merge \
  "$RES/entry_T0.5.jld2" "$RES/entry_T1.0.jld2" "$RES/entry_T2.0.jld2" "$RES/entry_T4.0.jld2" \
  "$RES/sim_info_gaussian_model.jld2"
echo "=== ALL_DONE exit=$? [$(date +%H:%M:%S)] ==="
