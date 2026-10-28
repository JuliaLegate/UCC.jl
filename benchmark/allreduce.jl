# Allreduce timing, one rank per GPU. Appends results to a CSV.
# Usage: mpiexec -n <gpus> julia --project allreduce.jl <backend> <out.csv>
#   backend: nccl (NCCL.jl), ucc_nccl (UCC over NCCL), ucc_ucp (UCC over UCX)

using MPI
using CUDA
using NCCL
using UCC

backend, outfile = ARGS

# UCC reads these when it initializes; `self` is only used on 1 GPU
if backend == "ucc_nccl"
    ENV["UCC_CL_BASIC_TLS"] = "self,ucp,nccl"
    ENV["UCC_TL_NCCL_TUNE"] = "cuda:inf"
elseif backend == "ucc_ucp"
    ENV["UCC_CL_BASIC_TLS"] = "self,ucp"
end

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)
CUDA.device!(rank % length(CUDA.devices()))

if backend == "nccl"
    id = MPI.bcast(rank == 0 ? NCCL.UniqueID() : nothing, comm; root=0)
    const nccl_comm = NCCL.Communicator(nranks, rank; unique_id=id)
    function allreduce!(x, y)
        NCCL.Allreduce!(x, y, +, nccl_comm)
        CUDA.synchronize()  # UCC.Allreduce! also blocks until done
    end
else
    const ctx = UCC.Context(UCC.MPIOOB(comm))
    const team = UCC.Team(ctx)
    allreduce!(x, y) = UCC.Allreduce!(x, y, +, team)
end

"Average seconds per allreduce of `nbytes`, on the slowest rank."
function time_allreduce(nbytes)
    x = CUDA.ones(Float32, nbytes ÷ 4)
    y = CUDA.zeros(Float32, nbytes ÷ 4)
    iters = nbytes <= 2^20 ? 100 : 20

    for _ in 1:5  # warmup
        allreduce!(x, y)
    end
    MPI.Barrier(comm)
    t = @elapsed for _ in 1:iters
        allreduce!(x, y)
    end

    @assert CUDA.@allowscalar(y[1]) == nranks
    CUDA.unsafe_free!(x)  # don't wait for GC; large sizes need the memory
    CUDA.unsafe_free!(y)
    return MPI.Allreduce(t / iters, max, comm)
end

if rank == 0 && !isfile(outfile)
    write(outfile, "backend,ngpus,bytes,time_us,busbw_GBps\n")
end

for nbytes in [2^k for k in 3:30]  # 8 B to 1 GiB
    t = time_allreduce(nbytes)
    busbw = nbytes / t * 2(nranks - 1) / nranks / 1e9  # nccl-tests definition
    if rank == 0
        open(io -> println(io, "$backend,$nranks,$nbytes,$(t * 1e6),$busbw"), outfile, "a")
        println("$backend  $nranks GPUs  $nbytes B  $(round(t * 1e6; digits=1)) us")
    end
end

if backend == "nccl"
    NCCL.destroy(nccl_comm)
else
    UCC.destroy(team)
    UCC.destroy(ctx)
end
MPI.Finalize()
