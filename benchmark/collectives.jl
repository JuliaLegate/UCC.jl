# Collective timing, one rank per GPU. Appends results to a CSV.
# Usage: mpiexec -n <gpus> julia --project=. collectives.jl <coll> <backend> <out.csv>
#            [--warmup=5] [--iters=N] [--no-barrier]
#   coll: allreduce, broadcast, reduce, allgather, reduce_scatter, alltoall,
#         gather, scatter (last two UCC only)
#   backend: nccl, ucc_nccl (UCC over NCCL), ucc_ucp (UCC over UCX)
#   --iters: timed calls per size (default 100 up to 1 MiB, 20 above)
#   --no-barrier: skip the untimed barrier between calls

using MPI
using CUDA
using NCCL
using UCC

coll, backend, outfile = ARGS[1:3]

warmup = 5
fixed_iters = nothing
barrier = true
for arg in ARGS[4:end]
    if startswith(arg, "--warmup=")
        global warmup = parse(Int, split(arg, '=')[2])
    elseif startswith(arg, "--iters=")
        global fixed_iters = parse(Int, split(arg, '=')[2])
    elseif arg == "--no-barrier"
        global barrier = false
    else
        error("unknown option $arg")
    end
end

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
else
    const ctx = UCC.Context(UCC.MPIOOB(comm))
    const team = UCC.Team(ctx)
end

# no ncclAllToAll; same as nccl-tests
function nccl_alltoall!(x, y)
    b = length(x) ÷ nranks
    NCCL.group() do
        for p in 0:(nranks - 1)
            NCCL.Send(view(x, (p * b + 1):((p + 1) * b)), nccl_comm; dest=p)
            NCCL.Recv!(view(y, (p * b + 1):((p + 1) * b)), nccl_comm; source=p)
        end
    end
end

const NCCL_CALLS = Dict(
    "allreduce" => (x, y) -> NCCL.Allreduce!(x, y, +, nccl_comm),
    "broadcast" => (x, y) -> NCCL.Broadcast!(x, y, nccl_comm; root=0),
    "reduce" => (x, y) -> NCCL.Reduce!(x, y, +, nccl_comm; root=0),
    "allgather" => (x, y) -> NCCL.Allgather!(x, y, nccl_comm),
    "reduce_scatter" => (x, y) -> NCCL.ReduceScatter!(x, y, +, nccl_comm),
    "alltoall" => nccl_alltoall!,
)

const UCC_CALLS = Dict(
    "allreduce" => (x, y) -> UCC.Allreduce!(x, y, +, team),
    "broadcast" => (x, y) -> UCC.Broadcast!(x, y, team; root=0),
    "reduce" => (x, y) -> UCC.Reduce!(x, y, +, team; root=0),
    "allgather" => (x, y) -> UCC.Allgather!(x, y, team),
    "reduce_scatter" => (x, y) -> UCC.ReduceScatter!(x, y, +, team),
    "alltoall" => (x, y) -> UCC.Alltoall!(x, y, team),
    "gather" => (x, y) -> UCC.Gather!(x, y, team; root=0),
    "scatter" => (x, y) -> UCC.Scatter!(x, y, team; root=0),
)

# nccl-tests bus bandwidth factors
const BUSBW_FACTOR = Dict(
    "allreduce" => n -> 2(n - 1) / n,
    "broadcast" => n -> 1,
    "reduce" => n -> 1,
    "allgather" => n -> (n - 1) / n,
    "reduce_scatter" => n -> (n - 1) / n,
    "alltoall" => n -> (n - 1) / n,
    "gather" => n -> (n - 1) / n,
    "scatter" => n -> (n - 1) / n,
)

const call! = backend == "nccl" ? NCCL_CALLS[coll] : UCC_CALLS[coll]

function run!(x, y)
    call!(x, y)
    backend == "nccl" && CUDA.synchronize()  # UCC calls already block
end

const r = Float32(rank + 1)
const n = Float32(nranks)
const isroot = rank == 0

# x, y for `count` elements in total (split into per-rank blocks where the
# collective does), and the expected (y[1], y[end]) on this rank, or `nothing`
function buffers(count)
    b = count ÷ nranks
    ones_(k) = CUDA.ones(Float32, k)
    zeros_(k) = CUDA.zeros(Float32, k)
    if coll == "allreduce"
        return ones_(count), zeros_(count), (n, n)
    elseif coll == "broadcast"
        return CUDA.fill(r, count), zeros_(count), (1, 1)
    elseif coll == "reduce"
        return ones_(count), zeros_(count), isroot ? (n, n) : nothing
    elseif coll == "allgather"
        return CUDA.fill(r, b), zeros_(b * nranks), (1, n)
    elseif coll == "reduce_scatter"
        return ones_(b * nranks), zeros_(b), (n, n)
    elseif coll == "alltoall"
        return CUDA.fill(r, b * nranks), zeros_(b * nranks), (1, n)
    elseif coll == "gather"
        y = isroot ? zeros_(b * nranks) : nothing
        return CUDA.fill(r, b), y, isroot ? (1, n) : nothing
    elseif coll == "scatter"  # block i of the root's buffer holds i
        x = isroot ? repeat(CuArray(Float32.(1:nranks)); inner=b) : nothing
        return x, zeros_(b), (r, r)
    end
    error("unknown collective $coll")
end

free(a) = a === nothing || CUDA.unsafe_free!(a)  # don't wait for GC

"Seconds per call of `count` elements: (min, avg, max) over ranks."
function time_coll(count)
    x, y, expect = buffers(count)
    iters = something(fixed_iters, count <= 2^18 ? 100 : 20)

    for _ in 1:warmup
        run!(x, y)
    end
    MPI.Barrier(comm)
    # as OSU: time the collective only, then barrier so senders (e.g. a
    # broadcast root) can't start the next call early
    t = 0.0
    for _ in 1:iters
        t_start = MPI.Wtime()
        run!(x, y)
        t += MPI.Wtime() - t_start
        barrier && MPI.Barrier(comm)
    end

    expect === nothing || @assert CUDA.@allowscalar((y[1], y[end])) == expect
    free(x)
    free(y)
    t /= iters
    tmin = MPI.Allreduce(t, min, comm)
    tmax = MPI.Allreduce(t, max, comm)
    tavg = MPI.Allreduce(t, +, comm) / nranks
    return tmin, tavg, tmax
end

if rank == 0 && !isfile(outfile)
    write(outfile, "coll,backend,ngpus,bytes,time_us,min_us,avg_us,busbw_GBps\n")
end

blocks = coll in ("allreduce", "broadcast", "reduce") ? 1 : nranks
for k in 3:30  # 8 B to 1 GiB
    count = (2^k ÷ 4) ÷ blocks * blocks
    count == 0 && continue
    tmin, tavg, tmax = time_coll(count)
    bytes = 4count
    busbw = bytes / tmax * BUSBW_FACTOR[coll](nranks) / 1e9  # slowest rank
    if rank == 0
        us = round.((tmax, tmin, tavg) .* 1e6; digits=2)
        open(io -> println(io, join((coll, backend, nranks, bytes, us..., busbw), ',')),
            outfile, "a")
        println("$coll  $backend  $nranks GPUs  $bytes B  $(us[1]) us")
    end
end

if backend == "nccl"
    NCCL.destroy(nccl_comm)
else
    UCC.destroy(team)
    UCC.destroy(ctx)
end
MPI.Finalize()
