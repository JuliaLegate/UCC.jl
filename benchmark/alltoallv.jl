# Alltoallv timing with variable per-peer message sizes, one rank per GPU.
# Appends results to a CSV.
# Usage: mpiexec -n <gpus> julia --project=. alltoallv.jl <backend> <out.csv>
#            [--dists=uniform,random,...] [--warmup=5] [--iters=N]
#            [--max-bytes=2^28] [--seed=1] [--no-barrier]
#   backend: nccl (grouped Send/Recv), ucc_nccl, ucc_ucp
#   --max-bytes: largest average send size per rank
#
# Distributions (weights W[src, dst], scaled to the average send size):
#   uniform  all pairs equal (= alltoall)
#   random   each pair uniform in [0, 2x mean]
#   zipf     sender sends 1/i to its i-th peer, peers randomly ordered
#   hotspot  rank 0 receives ~half of all traffic (incast)
#   sender   rank s sends s+1 parts to each peer (send imbalance)
#   sparse   each sender sends to 2 random peers (MoE top-2)

using MPI
using CUDA
using NCCL
using UCC
using Random

backend, outfile = ARGS[1:2]

dists = ["uniform", "random", "zipf", "hotspot", "sender", "sparse"]
warmup = 5
fixed_iters = nothing
max_bytes = 2^28
seed = 1
barrier = true
for arg in ARGS[3:end]
    key, val = occursin('=', arg) ? split(arg, '='; limit=2) : (arg, "")
    if key == "--dists"
        global dists = split(val, ',')
    elseif key == "--warmup"
        global warmup = parse(Int, val)
    elseif key == "--iters"
        global fixed_iters = parse(Int, val)
    elseif key == "--max-bytes"
        global max_bytes = Int(eval(Meta.parse(val)))  # e.g. 2^28
    elseif key == "--seed"
        global seed = parse(Int, val)
    elseif key == "--no-barrier"
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

function weights(dist, rng)
    n = nranks
    if dist == "uniform"
        return ones(n, n)
    elseif dist == "random"
        return 2 .* rand(rng, n, n)
    elseif dist == "zipf"
        W = zeros(n, n)
        for s in 1:n
            W[s, randperm(rng, n)] .= 1 ./ (1:n)
        end
        return W
    elseif dist == "hotspot"
        W = ones(n, n)
        W[:, 1] .= max(n - 1, 1)
        return W
    elseif dist == "sender"
        return repeat(Float64.(1:n), 1, n)
    elseif dist == "sparse"
        W = zeros(n, n)
        for s in 1:n
            W[s, randperm(rng, n)[1:min(2, n)]] .= 1
        end
        return W
    end
    error("unknown distribution $dist")
end

# C[src+1, dst+1], identical on all ranks
function counts_matrix(dist, count)
    C = Matrix{Int}(undef, nranks, nranks)
    if rank == 0
        W = weights(dist, Xoshiro(hash((seed, dist, nranks, count))))
        C .= floor.(Int, W .* (count * nranks / sum(W)))
    end
    return MPI.Bcast!(C, comm; root=0)
end

packed_displs(c) = cumsum(c) .- c

# no ncclAlltoallv: Send/Recv per nonempty pair in a group, as nccl-tests
function nccl_alltoallv!(x, sc, sd, y, rc, rd)
    NCCL.group() do
        for p in 0:(nranks - 1)
            i = p + 1
            sc[i] > 0 && NCCL.Send(view(x, sd[i] .+ (1:sc[i])), nccl_comm; dest=p)
            rc[i] > 0 && NCCL.Recv!(view(y, rd[i] .+ (1:rc[i])), nccl_comm; source=p)
        end
    end
    CUDA.synchronize()
end

function run!(x, sc, sd, y, rc, rd)
    if backend == "nccl"
        nccl_alltoallv!(x, sc, sd, y, rc, rd)
    else
        UCC.Alltoallv!(x, sc, y, rc, team; sdispls=sd, rdispls=rd)
    end
end

# value sent from rank s to rank d
tagval(s, d) = Float32(s * nranks + d + 1)

"Seconds per call: (min, avg, max) over ranks."
function time_alltoallv(C)
    sc = C[rank + 1, :]
    rc = C[:, rank + 1]
    sd = packed_displs(sc)
    rd = packed_displs(rc)
    # trailing pad: no empty CuArrays
    x = CuArray([reduce(vcat, [fill(tagval(rank, d), sc[d + 1]) for d in 0:(nranks - 1)]);
        0.0f0])
    y = CUDA.zeros(Float32, max(sum(rc), 1))
    iters = something(fixed_iters, sum(C) ÷ nranks <= 2^16 ? 100 : 20)

    for _ in 1:warmup
        run!(x, sc, sd, y, rc, rd)
    end
    MPI.Barrier(comm)
    t = 0.0
    for _ in 1:iters
        t_start = MPI.Wtime()
        run!(x, sc, sd, y, rc, rd)
        t += MPI.Wtime() - t_start
        barrier && MPI.Barrier(comm)
    end

    expect = reduce(vcat, [fill(tagval(s, rank), rc[s + 1]) for s in 0:(nranks - 1)];
        init=Float32[])
    @assert Array(y)[1:sum(rc)] == expect "wrong alltoallv result on rank $rank"
    CUDA.unsafe_free!(x)
    CUDA.unsafe_free!(y)
    t /= iters
    tmin = MPI.Allreduce(t, min, comm)
    tmax = MPI.Allreduce(t, max, comm)
    tavg = MPI.Allreduce(t, +, comm) / nranks
    return tmin, tavg, tmax
end

if rank == 0 && !isfile(outfile)
    write(outfile,
        "coll,dist,backend,ngpus,target_bytes,bytes,max_rank_bytes,imbalance,time_us,min_us,avg_us,busbw_GBps\n")
end

for dist in dists, k in 3:floor(Int, log2(max_bytes))
    count = 2^k ÷ 4
    C = counts_matrix(dist, count)
    sum(C) == 0 && continue
    tmin, tavg, tmax = time_alltoallv(C)
    # busiest rank's off-GPU send or receive bytes; busbw matches alltoall's for uniform
    offrank = [s == d ? 0 : C[s, d] for s in 1:nranks, d in 1:nranks]
    max_rank_bytes = 4 * maximum(max.(vec(sum(offrank; dims=2)), vec(sum(offrank; dims=1))))
    bytes = 4 * sum(C) ÷ nranks  # average send per rank
    imbalance = nranks == 1 ? 1.0 : max_rank_bytes / (bytes * (nranks - 1) / nranks)
    busbw = max_rank_bytes / tmax / 1e9
    if rank == 0
        us = round.((tmax, tmin, tavg) .* 1e6; digits=2)
        row = ("alltoallv", dist, backend, nranks, 4count, bytes, max_rank_bytes,
            round(imbalance; digits=3), us..., busbw)
        open(io -> println(io, join(row, ',')), outfile, "a")
        println("alltoallv  $dist  $backend  $nranks GPUs  $bytes B  $(us[1]) us")
    end
end

if backend == "nccl"
    NCCL.destroy(nccl_comm)
else
    UCC.destroy(team)
    UCC.destroy(ctx)
end
MPI.Finalize()
