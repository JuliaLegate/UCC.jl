#!/usr/bin/env bash
# Allreduce on 1..N GPUs (default: all) with NCCL, UCC+NCCL, UCC+UCP; then plot.
set -e
cd "$(dirname "$0")"
N=${1:-$(nvidia-smi -L | wc -l)}
export UCX_TLS=${UCX_TLS:-^tcp} # UCX's TCP transport can hang on startup
export JULIA_CUDA_MEMORY_POOL=none
out=results.csv
rm -f $out

julia --project=. -e 'using Pkg; Pkg.instantiate()'
for n in $(seq 1 "$N"); do
    for b in nccl ucc_nccl ucc_ucp; do
        julia --project=. -e "using MPI; run(\`\$(mpiexec()) -n $n \$(Base.julia_cmd()) --project=. allreduce.jl $b $out\`)"
    done
done
julia --project=. plot.jl $out
