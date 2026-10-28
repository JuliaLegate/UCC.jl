# Collectives
#
# Every collective comes in a blocking form (`Allreduce!`, returning the receive
# buffer, like NCCL.jl) and a nonblocking form (`Iallreduce!`, returning a
# `Request`). The nonblocking forms accept these extra keywords:
#
# - `post=true`: set to `false` to only initialize the request (post it later
#   with `UCC.post!`, or trigger it on an execution engine)
# - `persistent=false`: allow the request to be re-posted after completion
# - `timeout`: fail with `UCC_ERR_TIMED_OUT` after this many seconds
# - `tag`: collective tag used for ordering

_datatype(::Type{T}) where {T} = get(() -> _builtin_datatype(T), _generic_datatypes, T)

function _info(buffer, count::Integer=length(buffer))
    _check_contiguous(buffer)
    ucc_coll_buffer_info_u(ucc_coll_buffer_info_t(
        _ptr(buffer), count, _datatype(eltype(buffer)), memory_type(buffer)))
end

# placeholder for a buffer that is not used on this rank (e.g. the receive
# buffer of a non-root rank)
_info(::Nothing, like) = ucc_coll_buffer_info_u(ucc_coll_buffer_info_t(
    C_NULL, 0, _datatype(eltype(like)), memory_type(like)))

function _info_v(buffer, counts::Vector{UInt64}, displs::Vector{UInt64})
    _check_contiguous(buffer)
    ucc_coll_buffer_info_u(ucc_coll_buffer_info_v_t(
        _ptr(buffer), pointer(counts), pointer(displs),
        _datatype(eltype(buffer)), memory_type(buffer)))
end

function _counts(counts, team::Team)
    length(counts) == team.size || throw(DimensionMismatch(
        "expected $(team.size) counts, got $(length(counts))"))
    return UInt64[c for c in counts]
end

# default displacements: blocks packed back to back
function _displs(::Nothing, counts::Vector{UInt64})
    displs = similar(counts)
    offset = UInt64(0)
    for i in eachindex(counts)
        displs[i] = offset
        offset = Base.Checked.checked_add(offset, counts[i])
    end
    return displs
end

function _displs(displs, counts::Vector{UInt64})
    length(displs) == length(counts) || throw(DimensionMismatch(
        "expected $(length(counts)) displacements, got $(length(displs))"))
    return UInt64[d for d in displs]
end

function _check_extent(buffer, counts, displs, name)
    n = UInt64(length(buffer))
    for (count, offset) in zip(counts, displs)
        count == 0 && continue
        offset <= n && count <= n - offset || throw(DimensionMismatch(
            "$name is too short for count $count at displacement $offset"))
    end
    return nothing
end

# UCC 1.8 can complete a reduction with an unsupported (type, op) pair
# without reporting an error, so reject those up front.
_op_supported(::Type, o::ucc_reduction_op_t) = o != UCC_OP_MAXLOC && o != UCC_OP_MINLOC
_op_supported(::Type{<:AbstractFloat}, o::ucc_reduction_op_t) =  # includes Core.BFloat16 on Julia 1.11+
    o in (UCC_OP_SUM, UCC_OP_PROD, UCC_OP_MAX, UCC_OP_MIN, UCC_OP_AVG)
_op_supported(::Type{<:Complex}, o::ucc_reduction_op_t) = o in (UCC_OP_SUM, UCC_OP_PROD, UCC_OP_AVG)
_op_supported(::Type{Bool}, o::ucc_reduction_op_t) = o ∉ (UCC_OP_SUM, UCC_OP_PROD, UCC_OP_AVG, UCC_OP_MAXLOC, UCC_OP_MINLOC)
_op_supported(::Type{<:Union{Int128,UInt128}}, ::ucc_reduction_op_t) = false
_op_supported(T, o, mem) = _op_supported(T, o)
_op_supported(::Type{Float16}, o, mem) =  # no CPU reduction
    mem in (UCC_MEMORY_TYPE_CUDA, UCC_MEMORY_TYPE_CUDA_MANAGED) && _op_supported(AbstractFloat, o)

function _op(buffer, op)
    T = eltype(buffer)
    o = ucc_reduction_op_t(op)
    if haskey(_generic_datatype_objs, T)  # reduction defined by the datatype
        _generic_datatype_objs[T].op === nothing && throw(ArgumentError("$T was registered without an op"))
        return o
    end
    _op_supported(T, o, memory_type(buffer)) ||
        throw(ArgumentError("reduction $o is not supported for element type $T"))
    return o
end

function _check_eltypes(buffers...)
    T = eltype(first(buffers))
    for b in buffers
        b === nothing && continue
        eltype(b) == T || throw(ArgumentError(
            "all buffers must have the same element type (got $T and $(eltype(b)))"))
    end
end

function _check_length(buffer, n, name)
    length(buffer) == n || throw(DimensionMismatch(
        "$name has length $(length(buffer)), expected $n"))
end

_isroot(team::Team, root) = team.rank == root

#
# Allreduce
#

"""
    UCC.Allreduce!(sendbuf, recvbuf, op, team::Team)
    UCC.Iallreduce!(sendbuf, recvbuf, op, team::Team; kwargs...) :: Request

Reduce array `sendbuf` using `op` (one of `+`, `*`, `min`, `max`, `&`, `|`,
`xor`, [`UCC.avg`](@ref), [`UCC.land`](@ref), …), writing the result to
`recvbuf` on all ranks.

# External links
- `UCC_COLL_TYPE_ALLREDUCE`
"""
function Iallreduce!(sendbuf, recvbuf, op, team::Team; kwargs...)
    _check_length(sendbuf, length(recvbuf), "sendbuf")
    _check_eltypes(sendbuf, recvbuf)
    _collective(team, UCC_COLL_TYPE_ALLREDUCE;
                src=_info(sendbuf), dst=_info(recvbuf), op=_op(sendbuf, op),
                buffers=(sendbuf, recvbuf), result=recvbuf, kwargs...)
end

"""
    UCC.Allreduce!(sendrecvbuf, op, team::Team)
    UCC.Iallreduce!(sendrecvbuf, op, team::Team; kwargs...) :: Request

Reduce the array `sendrecvbuf` using `op`, writing the result in place on all
ranks.
"""
function Iallreduce!(sendrecvbuf, op, team::Team; kwargs...)
    _collective(team, UCC_COLL_TYPE_ALLREDUCE;
                src=_info(sendrecvbuf), dst=_info(sendrecvbuf), op=_op(sendrecvbuf, op),
                flags=UCC_COLL_ARGS_FLAG_IN_PLACE,
                buffers=(sendrecvbuf,), result=sendrecvbuf, kwargs...)
end

#
# Broadcast
#

"""
    UCC.Broadcast!(sendrecvbuf, team::Team; root=0)
    UCC.Ibroadcast!(sendrecvbuf, team::Team; root=0, kwargs...) :: Request

Copy `sendrecvbuf` on rank `root` to `sendrecvbuf` on all ranks.

# External links
- `UCC_COLL_TYPE_BCAST`
"""
function Ibroadcast!(sendrecvbuf, team::Team; root::Integer=0, kwargs...)
    _collective(team, UCC_COLL_TYPE_BCAST; src=_info(sendrecvbuf), root,
                buffers=(sendrecvbuf,), result=sendrecvbuf, kwargs...)
end

"""
    UCC.Broadcast!(sendbuf, recvbuf, team::Team; root=0)
    UCC.Ibroadcast!(sendbuf, recvbuf, team::Team; root=0, kwargs...) :: Request

Copy `sendbuf` on rank `root` to `recvbuf` on all ranks. `sendbuf` is only
used on `root`, and may be `nothing` elsewhere.

The two-buffer form requires immediate, non-persistent posting. For
`persistent=true` or `post=false` (including triggered posting), use
`Ibroadcast!(recvbuf, team; ...)` and copy into `recvbuf` on the root before
each post, after the previous operation has completed.
"""
function Ibroadcast!(sendbuf, recvbuf, team::Team; root::Integer=0,
                     post::Bool=true, persistent::Bool=false, kwargs...)
    (persistent || !post) && throw(ArgumentError(
        "two-buffer broadcast does not support persistent=true or post=false; " *
        "use Ibroadcast!(recvbuf, team; ...) and copyto!(recvbuf, sendbuf) on " *
        "the root before each post, after the previous operation has completed"))
    if _isroot(team, root) && sendbuf !== recvbuf
        _check_length(sendbuf, length(recvbuf), "sendbuf")
        copyto!(recvbuf, sendbuf)
    end
    Ibroadcast!(recvbuf, team; root, kwargs...)
end

#
# Reduce
#

"""
    UCC.Reduce!(sendbuf, recvbuf, op, team::Team; root=0)
    UCC.Ireduce!(sendbuf, recvbuf, op, team::Team; root=0, kwargs...) :: Request

Reduce the array `sendbuf` using `op`, writing the result to `recvbuf` on rank
`root`. `recvbuf` is only used on `root`, and may be `nothing` elsewhere.

# External links
- `UCC_COLL_TYPE_REDUCE`
"""
function Ireduce!(sendbuf, recvbuf, op, team::Team; root::Integer=0, kwargs...)
    if _isroot(team, root)
        _check_length(recvbuf, length(sendbuf), "recvbuf")
        _check_eltypes(sendbuf, recvbuf)
        dst = _info(recvbuf)
    else
        dst = recvbuf === nothing ? _info(nothing, sendbuf) : _info(recvbuf)
    end
    _collective(team, UCC_COLL_TYPE_REDUCE;
                src=_info(sendbuf), dst, op=_op(sendbuf, op), root,
                buffers=(sendbuf, recvbuf), result=recvbuf, kwargs...)
end

"""
    UCC.Reduce!(sendrecvbuf, op, team::Team; root=0)
    UCC.Ireduce!(sendrecvbuf, op, team::Team; root=0, kwargs...) :: Request

Reduce the array `sendrecvbuf` using `op`, writing the result in place on rank
`root`.
"""
function Ireduce!(sendrecvbuf, op, team::Team; root::Integer=0, kwargs...)
    flags = _isroot(team, root) ? UCC_COLL_ARGS_FLAG_IN_PLACE : UInt64(0)
    _collective(team, UCC_COLL_TYPE_REDUCE;
                src=_info(sendrecvbuf), dst=_info(sendrecvbuf),
                op=_op(sendrecvbuf, op), root, flags,
                buffers=(sendrecvbuf,), result=sendrecvbuf, kwargs...)
end

#
# Allgather(v)
#

"""
    UCC.Allgather!(sendbuf, recvbuf, team::Team)
    UCC.Iallgather!(sendbuf, recvbuf, team::Team; kwargs...) :: Request

Concatenate `sendbuf` from each rank into `recvbuf` on all ranks.

# External links
- `UCC_COLL_TYPE_ALLGATHER`
"""
function Iallgather!(sendbuf, recvbuf, team::Team; kwargs...)
    _check_length(recvbuf, length(sendbuf) * team.size, "recvbuf")
    _check_eltypes(sendbuf, recvbuf)
    _collective(team, UCC_COLL_TYPE_ALLGATHER; src=_info(sendbuf), dst=_info(recvbuf),
                buffers=(sendbuf, recvbuf), result=recvbuf, kwargs...)
end

"""
    UCC.Allgather!(sendrecvbuf, team::Team)
    UCC.Iallgather!(sendrecvbuf, team::Team; kwargs...) :: Request

In-place allgather: each rank's contribution is the `rank`-th block of
`sendrecvbuf`.
"""
function Iallgather!(sendrecvbuf, team::Team; kwargs...)
    length(sendrecvbuf) % team.size == 0 || throw(DimensionMismatch(
        "length of sendrecvbuf must be a multiple of the team size"))
    _collective(team, UCC_COLL_TYPE_ALLGATHER;
                src=_info(sendrecvbuf), dst=_info(sendrecvbuf),
                flags=UCC_COLL_ARGS_FLAG_IN_PLACE,
                buffers=(sendrecvbuf,), result=sendrecvbuf, kwargs...)
end

"""
    UCC.Allgatherv!(sendbuf, recvbuf, counts, team::Team; displs=nothing)
    UCC.Iallgatherv!(sendbuf, recvbuf, counts, team::Team; displs=nothing, kwargs...) :: Request

Gather `sendbuf` from every rank into `recvbuf` on all ranks, where rank `i`
contributes `counts[i+1]` elements placed at element offset `displs[i+1]`
(blocks packed back to back by default).

# External links
- `UCC_COLL_TYPE_ALLGATHERV`
"""
function Iallgatherv!(sendbuf, recvbuf, counts, team::Team; displs=nothing, kwargs...)
    c = _counts(counts, team)
    d = _displs(displs, c)
    _check_length(sendbuf, c[team.rank+1], "sendbuf")
    _check_eltypes(sendbuf, recvbuf)
    _check_extent(recvbuf, c, d, "recvbuf")
    _collective(team, UCC_COLL_TYPE_ALLGATHERV;
                src=_info(sendbuf), dst=_info_v(recvbuf, c, d),
                buffers=(sendbuf, recvbuf, c, d), result=recvbuf, kwargs...)
end

#
# Alltoall(v)
#

"""
    UCC.Alltoall!(sendbuf, recvbuf, team::Team)
    UCC.Ialltoall!(sendbuf, recvbuf, team::Team; kwargs...) :: Request

Send the `i`-th block of `sendbuf` to rank `i`, receiving the `j`-th block of
`recvbuf` from rank `j`. Both buffers have `size(team)` equal blocks.

# External links
- `UCC_COLL_TYPE_ALLTOALL`
"""
function Ialltoall!(sendbuf, recvbuf, team::Team; kwargs...)
    _check_length(recvbuf, length(sendbuf), "recvbuf")
    length(sendbuf) % team.size == 0 || throw(DimensionMismatch(
        "length of sendbuf must be a multiple of the team size"))
    _check_eltypes(sendbuf, recvbuf)
    _collective(team, UCC_COLL_TYPE_ALLTOALL; src=_info(sendbuf), dst=_info(recvbuf),
                buffers=(sendbuf, recvbuf), result=recvbuf, kwargs...)
end

"""
    UCC.Alltoall!(sendrecvbuf, team::Team)
    UCC.Ialltoall!(sendrecvbuf, team::Team; kwargs...) :: Request

In-place all-to-all.
"""
function Ialltoall!(sendrecvbuf, team::Team; kwargs...)
    length(sendrecvbuf) % team.size == 0 || throw(DimensionMismatch(
        "length of sendrecvbuf must be a multiple of the team size"))
    _collective(team, UCC_COLL_TYPE_ALLTOALL;
                src=_info(sendrecvbuf), dst=_info(sendrecvbuf),
                flags=UCC_COLL_ARGS_FLAG_IN_PLACE,
                buffers=(sendrecvbuf,), result=sendrecvbuf, kwargs...)
end

"""
    UCC.Alltoallv!(sendbuf, sendcounts, recvbuf, recvcounts, team::Team;
                   sdispls=nothing, rdispls=nothing)
    UCC.Ialltoallv!(sendbuf, sendcounts, recvbuf, recvcounts, team::Team;
                    sdispls=nothing, rdispls=nothing, kwargs...) :: Request

Send `sendcounts[i+1]` elements at offset `sdispls[i+1]` of `sendbuf` to rank
`i`, and receive `recvcounts[j+1]` elements at offset `rdispls[j+1]` of
`recvbuf` from rank `j`. Displacements default to packed blocks.

# External links
- `UCC_COLL_TYPE_ALLTOALLV`
"""
function Ialltoallv!(sendbuf, sendcounts, recvbuf, recvcounts, team::Team;
                     sdispls=nothing, rdispls=nothing, kwargs...)
    sc = _counts(sendcounts, team)
    rc = _counts(recvcounts, team)
    sd = _displs(sdispls, sc)
    rd = _displs(rdispls, rc)
    _check_eltypes(sendbuf, recvbuf)
    _check_extent(sendbuf, sc, sd, "sendbuf")
    _check_extent(recvbuf, rc, rd, "recvbuf")
    _collective(team, UCC_COLL_TYPE_ALLTOALLV;
                src=_info_v(sendbuf, sc, sd), dst=_info_v(recvbuf, rc, rd),
                buffers=(sendbuf, recvbuf, sc, sd, rc, rd), result=recvbuf, kwargs...)
end

#
# Barrier, fan-in, fan-out
#

"""
    UCC.Barrier(team::Team)
    UCC.Ibarrier(team::Team; kwargs...) :: Request

Block until all ranks of `team` have reached the barrier.

# External links
- `UCC_COLL_TYPE_BARRIER`
"""
Ibarrier(team::Team; kwargs...) = _collective(team, UCC_COLL_TYPE_BARRIER; kwargs...)

"""
    UCC.Fanin(team::Team; root=0)
    UCC.Ifanin(team::Team; root=0, kwargs...) :: Request

Rank `root` waits until all ranks have entered the fan-in.

# External links
- `UCC_COLL_TYPE_FANIN`
"""
Ifanin(team::Team; root::Integer=0, kwargs...) =
    _collective(team, UCC_COLL_TYPE_FANIN; root, kwargs...)

"""
    UCC.Fanout(team::Team; root=0)
    UCC.Ifanout(team::Team; root=0, kwargs...) :: Request

All ranks wait until rank `root` has entered the fan-out.

# External links
- `UCC_COLL_TYPE_FANOUT`
"""
Ifanout(team::Team; root::Integer=0, kwargs...) =
    _collective(team, UCC_COLL_TYPE_FANOUT; root, kwargs...)

#
# Gather(v)
#

"""
    UCC.Gather!(sendbuf, recvbuf, team::Team; root=0)
    UCC.Igather!(sendbuf, recvbuf, team::Team; root=0, kwargs...) :: Request

Concatenate `sendbuf` from each rank into `recvbuf` on rank `root`. `recvbuf`
is only used on `root`, and may be `nothing` elsewhere.

# External links
- `UCC_COLL_TYPE_GATHER`
"""
function Igather!(sendbuf, recvbuf, team::Team; root::Integer=0, kwargs...)
    if _isroot(team, root)
        _check_length(recvbuf, length(sendbuf) * team.size, "recvbuf")
        _check_eltypes(sendbuf, recvbuf)
        dst = _info(recvbuf)
    else
        dst = _info(nothing, sendbuf)
    end
    _collective(team, UCC_COLL_TYPE_GATHER; src=_info(sendbuf), dst, root,
                buffers=(sendbuf, recvbuf), result=recvbuf, kwargs...)
end

"""
    UCC.Gatherv!(sendbuf, recvbuf, counts, team::Team; root=0, displs=nothing)
    UCC.Igatherv!(sendbuf, recvbuf, counts, team::Team; root=0, displs=nothing, kwargs...) :: Request

Gather `sendbuf` (of length `counts[rank+1]`) from each rank into `recvbuf` on
rank `root`, at element offsets `displs` (packed by default). `recvbuf` may be
`nothing` on non-root ranks.

# External links
- `UCC_COLL_TYPE_GATHERV`
"""
function Igatherv!(sendbuf, recvbuf, counts, team::Team; root::Integer=0,
                   displs=nothing, kwargs...)
    c = _counts(counts, team)
    d = _displs(displs, c)
    _check_length(sendbuf, c[team.rank+1], "sendbuf")
    if _isroot(team, root)
        _check_eltypes(sendbuf, recvbuf)
        _check_extent(recvbuf, c, d, "recvbuf")
        dst = _info_v(recvbuf, c, d)
    else
        dst = _info_v(nothing, sendbuf, c, d)
    end
    _collective(team, UCC_COLL_TYPE_GATHERV; src=_info(sendbuf), dst, root,
                buffers=(sendbuf, recvbuf, c, d), result=recvbuf, kwargs...)
end

_info_v(::Nothing, like, counts::Vector{UInt64}, displs::Vector{UInt64}) =
    ucc_coll_buffer_info_u(ucc_coll_buffer_info_v_t(
        C_NULL, pointer(counts), pointer(displs),
        _datatype(eltype(like)), memory_type(like)))

#
# Reduce-scatter(v)
#

"""
    UCC.ReduceScatter!(sendbuf, recvbuf, op, team::Team)
    UCC.IreduceScatter!(sendbuf, recvbuf, op, team::Team; kwargs...) :: Request

Reduce `sendbuf` from each rank using `op`, and leave the reduced result
scattered over the ranks such that `recvbuf` on rank `i` contains the `i`-th
block of the result.

# External links
- `UCC_COLL_TYPE_REDUCE_SCATTER`
"""
function IreduceScatter!(sendbuf, recvbuf, op, team::Team; kwargs...)
    _check_length(sendbuf, length(recvbuf) * team.size, "sendbuf")
    _check_eltypes(sendbuf, recvbuf)
    _collective(team, UCC_COLL_TYPE_REDUCE_SCATTER;
                src=_info(sendbuf), dst=_info(recvbuf), op=_op(sendbuf, op),
                buffers=(sendbuf, recvbuf), result=recvbuf, kwargs...)
end

"""
    UCC.ReduceScatterv!(sendbuf, recvbuf, counts, op, team::Team)
    UCC.IreduceScatterv!(sendbuf, recvbuf, counts, op, team::Team; kwargs...) :: Request

Reduce `sendbuf` (of length `sum(counts)`) from each rank using `op`; rank `i`
receives the `counts[i+1]` elements of the result starting at offset
`sum(counts[1:i])` in `recvbuf`.

# External links
- `UCC_COLL_TYPE_REDUCE_SCATTERV`
"""
function IreduceScatterv!(sendbuf, recvbuf, counts, op, team::Team; kwargs...)
    c = _counts(counts, team)
    d = _displs(nothing, c)
    _check_length(sendbuf, sum(c), "sendbuf")
    _check_length(recvbuf, c[team.rank+1], "recvbuf")
    _check_eltypes(sendbuf, recvbuf)
    _collective(team, UCC_COLL_TYPE_REDUCE_SCATTERV;
                src=_info(sendbuf), dst=_info_v(recvbuf, c, d),
                op=_op(sendbuf, op),
                buffers=(sendbuf, recvbuf, c, d), result=recvbuf, kwargs...)
end

#
# Scatter(v)
#

"""
    UCC.Scatter!(sendbuf, recvbuf, team::Team; root=0)
    UCC.Iscatter!(sendbuf, recvbuf, team::Team; root=0, kwargs...) :: Request

Send the `i`-th block of `sendbuf` on rank `root` to `recvbuf` on rank `i`.
`sendbuf` is only used on `root`, and may be `nothing` elsewhere.

UCC 1.8's transports do not implement `UCC_COLL_TYPE_SCATTER`, so this is
posted as a scatterv with equal counts.
"""
function Iscatter!(sendbuf, recvbuf, team::Team; root::Integer=0, kwargs...)
    _isroot(team, root) && _check_length(sendbuf, length(recvbuf) * team.size, "sendbuf")
    Iscatterv!(sendbuf, fill(length(recvbuf), team.size), recvbuf, team; root, kwargs...)
end

"""
    UCC.Scatterv!(sendbuf, counts, recvbuf, team::Team; root=0, displs=nothing)
    UCC.Iscatterv!(sendbuf, counts, recvbuf, team::Team; root=0, displs=nothing, kwargs...) :: Request

Send `counts[i+1]` elements at offset `displs[i+1]` (packed by default) of
`sendbuf` on rank `root` to `recvbuf` on rank `i`. `sendbuf` may be `nothing`
on non-root ranks.

# External links
- `UCC_COLL_TYPE_SCATTERV`
"""
function Iscatterv!(sendbuf, counts, recvbuf, team::Team; root::Integer=0,
                    displs=nothing, kwargs...)
    c = _counts(counts, team)
    d = _displs(displs, c)
    _check_length(recvbuf, c[team.rank+1], "recvbuf")
    if _isroot(team, root)
        _check_eltypes(sendbuf, recvbuf)
        _check_extent(sendbuf, c, d, "sendbuf")
        src = _info_v(sendbuf, c, d)
    else
        src = _info_v(nothing, recvbuf, c, d)
    end
    _collective(team, UCC_COLL_TYPE_SCATTERV; src, dst=_info(recvbuf), root,
                buffers=(sendbuf, recvbuf, c, d), result=recvbuf, kwargs...)
end

#
# Blocking variants
#

for (blocking, nonblocking) in [(:Allreduce!, :Iallreduce!),
                                (:Broadcast!, :Ibroadcast!),
                                (:Reduce!, :Ireduce!),
                                (:Allgather!, :Iallgather!),
                                (:Allgatherv!, :Iallgatherv!),
                                (:Alltoall!, :Ialltoall!),
                                (:Alltoallv!, :Ialltoallv!),
                                (:Barrier, :Ibarrier),
                                (:Fanin, :Ifanin),
                                (:Fanout, :Ifanout),
                                (:Gather!, :Igather!),
                                (:Gatherv!, :Igatherv!),
                                (:ReduceScatter!, :IreduceScatter!),
                                (:ReduceScatterv!, :IreduceScatterv!),
                                (:Scatter!, :Iscatter!),
                                (:Scatterv!, :Iscatterv!)]
    @eval begin
        $blocking(args...; kwargs...) = _blocking($nonblocking(args...; kwargs...))
        @doc (@doc $nonblocking) $blocking
    end
end
