#!/usr/bin/env bash
# Alltoallv on N GPUs (default: all), one launch per backend; then plot.
# Usage: ./run_alltoallv.sh [N] [dist...]   (default: all distributions)
# WARMUP, ITERS, MAX_BYTES (e.g. 2^24) and BACKENDS env vars are passed through.
set -e
cd "$(dirname "$0")"
N=${1:-$(nvidia-smi -L | wc -l)}
shift || true
dists=$(echo ${*:-uniform random zipf hotspot sender sparse} | tr ' ' ,)
backends=${BACKENDS:-nccl ucc_nccl ucc_ucp}
export UCX_TLS=${UCX_TLS:-^tcp} # UCX's TCP transport can hang on startup
export JULIA_CUDA_MEMORY_POOL=none
out=alltoallv.csv
opts="${WARMUP:+--warmup=$WARMUP} ${ITERS:+--iters=$ITERS} ${MAX_BYTES:+--max-bytes=$MAX_BYTES}"
rm -f $out

julia --project=. -e 'using Pkg; Pkg.instantiate()'
for b in $backends; do
    julia --project=. -e "using MPI; run(\`\$(mpiexec()) -n $N \$(Base.julia_cmd()) --project=. alltoallv.jl $b $out --dists=$dists $opts\`)" ||
        echo "alltoallv $b on $N GPUs failed; continuing"
done
julia --project=. plot_alltoallv.jl $out
