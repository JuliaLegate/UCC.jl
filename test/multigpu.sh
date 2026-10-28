#!/usr/bin/env bash
# Run mpi_collectives.jl with one rank per local GPU, once per UCC transport.
set -e
cd "$(dirname "$0")"
n=${1:-$(nvidia-smi -L | wc -l)}
export UCC_TEST_CUDA=true UCX_TLS=${UCX_TLS:-^tcp} # UCX's TCP transport can hang on startup
export JULIA_CUDA_MEMORY_POOL=none # TL/CUDA can't share pool allocations over IPC

julia --project=. -e 'using Pkg; Pkg.instantiate()'
mpi() { julia --project=. -e "using MPI; run(\`\$(mpiexec()) -n $n \$(Base.julia_cmd()) --project=. mpi_collectives.jl\`)"; }

mpi                                                     # UCC's choice
UCC_CL_BASIC_TLS=ucp,nccl UCC_TL_NCCL_TUNE=cuda:inf mpi # force NCCL
UCC_CL_BASIC_TLS=ucp,cuda UCC_TL_CUDA_TUNE=cuda:inf mpi # force TL/CUDA; falls back to UCP if GPUs lack NVLink
