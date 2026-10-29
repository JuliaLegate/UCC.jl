#!/usr/bin/env bash
# Each collective on 1..N GPUs (default: all) with NCCL, UCC+NCCL, UCC+UCP; then plot.
# Usage: ./run.sh [N] [collective...]   (default: all collectives)
# WARMUP and ITERS env vars are passed to collectives.jl.
set -e
cd "$(dirname "$0")"
N=${1:-$(nvidia-smi -L | wc -l)}
shift || true
colls=${*:-allreduce broadcast reduce allgather reduce_scatter alltoall gather scatter}
export UCX_TLS=${UCX_TLS:-^tcp} # UCX's TCP transport can hang on startup
export JULIA_CUDA_MEMORY_POOL=none
out=results.csv
opts="${WARMUP:+--warmup=$WARMUP} ${ITERS:+--iters=$ITERS}"
rm -f $out

julia --project=. -e 'using Pkg; Pkg.instantiate()'
for c in $colls; do
    for n in $(seq 1 "$N"); do
        for b in nccl ucc_nccl ucc_ucp; do
            [[ $b == nccl && ($c == gather || $c == scatter) ]] && continue # not in NCCL
            julia --project=. -e "using MPI; run(\`\$(mpiexec()) -n $n \$(Base.julia_cmd()) --project=. collectives.jl $c $b $out $opts\`)" ||
                echo "$c $b on $n GPUs failed; continuing"
        done
    done
done
julia --project=. plot.jl $out
