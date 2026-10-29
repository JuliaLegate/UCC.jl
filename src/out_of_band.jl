# Out-of-band (OOB) communication
#
# UCC bootstraps contexts and teams with a user-provided allgather. Julia OOB
# implementations subtype `AbstractOOB` and are called back from UCC through the
# C trampolines below.

export AbstractOOB, LocalOOB

"""
    UCC.AbstractOOB

Out-of-band transport used by UCC to bootstrap [`Context`](@ref)s and
[`Team`](@ref)s. Subtypes must be `mutable` (UCC holds a pointer to them) and
implement

- `UCC.oob_rank(oob) :: Integer` – 0-based index of the calling process,
- `UCC.oob_size(oob) :: Integer` – number of participating processes,

and either the blocking

- `UCC.allgather(oob, data::Vector{UInt8}) :: Vector{UInt8}` – concatenation of
  `data` from all ranks, in rank order,

or the nonblocking pair

- `UCC.allgather_start(oob, src::Ptr{UInt8}, dst::Ptr{UInt8}, nbytes) -> handle`
- `UCC.allgather_test(oob, handle) :: Bool`

(plus optionally `UCC.allgather_free(oob, handle)`).

The MPI extension (`using MPI`) provides `UCC.MPIOOB(comm)`.
"""
abstract type AbstractOOB end

function oob_rank end
function oob_size end

function allgather(oob::AbstractOOB, data::Vector{UInt8})
    return throw(MethodError(allgather, (oob, data)))
end

function allgather_start(oob::AbstractOOB, src::Ptr{UInt8}, dst::Ptr{UInt8}, nbytes::Int)
    data = copy(unsafe_wrap(Array, src, nbytes))
    result = allgather(oob, data)
    length(result) == nbytes * oob_size(oob) || error(
        "OOB allgather returned $(length(result)) bytes, expected $(nbytes * oob_size(oob))")
    GC.@preserve result unsafe_copyto!(dst, pointer(result), length(result))
    return nothing
end
allgather_test(oob::AbstractOOB, handle) = true
allgather_free(oob::AbstractOOB, handle) = nothing

# Outstanding OOB requests, keyed by the integer id UCC sees as `void *request`.
mutable struct OOBRequest
    oob::AbstractOOB
    handle::Any
    done::Bool
end

const _oob_requests = Dict{UInt,OOBRequest}()
const _oob_requests_lock = ReentrantLock()
const _oob_next_id = Ref{UInt}(1)

_oob_request(request::Ptr{Cvoid}) = @lock _oob_requests_lock _oob_requests[UInt(request)]

function _oob_allgather(src::Ptr{Cvoid}, dst::Ptr{Cvoid}, nbytes::Csize_t,
    info::Ptr{Cvoid}, request::Ptr{Ptr{Cvoid}})::ucc_status_t
    try
        oob = unsafe_pointer_to_objref(info)::AbstractOOB
        handle = allgather_start(oob, Ptr{UInt8}(src), Ptr{UInt8}(dst), Int(nbytes))
        id = @lock _oob_requests_lock begin
            id = _oob_next_id[]
            _oob_next_id[] += 1
            _oob_requests[id] = OOBRequest(oob, handle, false)
            id
        end
        unsafe_store!(request, Ptr{Cvoid}(id))
        return UCC_OK
    catch err
        @error "UCC OOB allgather failed" exception = (err, catch_backtrace())
        return UCC_ERR_NO_MESSAGE
    end
end

function _oob_req_test(request::Ptr{Cvoid})::ucc_status_t
    try
        req = _oob_request(request)
        req.done || (req.done = allgather_test(req.oob, req.handle))
        req.done && return UCC_OK
        yield()  # let other tasks (e.g. in-process ranks) make progress
        return UCC_INPROGRESS
    catch err
        @error "UCC OOB request test failed" exception = (err, catch_backtrace())
        return UCC_ERR_NO_MESSAGE
    end
end

function _oob_req_free(request::Ptr{Cvoid})::ucc_status_t
    try
        req = @lock _oob_requests_lock pop!(_oob_requests, UInt(request))
        allgather_free(req.oob, req.handle)
        return UCC_OK
    catch err
        @error "UCC OOB request free failed" exception = (err, catch_backtrace())
        return UCC_ERR_NO_MESSAGE
    end
end

# Build the C `ucc_oob_coll_t` for `oob`. The caller must keep `oob` rooted for
# as long as UCC may call back into it.
function _oob_coll(oob::AbstractOOB)
    ismutable(oob) || throw(ArgumentError("OOB objects must be mutable (got $(typeof(oob)))"))
    return LibUCC.ucc_oob_coll_t(
        @cfunction(_oob_allgather, ucc_status_t,
            (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Ptr{Ptr{Cvoid}})),
        @cfunction(_oob_req_test, ucc_status_t, (Ptr{Cvoid},)),
        @cfunction(_oob_req_free, ucc_status_t, (Ptr{Cvoid},)),
        pointer_from_objref(oob),
        UInt32(oob_size(oob)),
        UInt32(oob_rank(oob)))
end

mutable struct LocalOOBRound
    data::Vector{Vector{UInt8}}
    arrived::Int
    collected::Int
end

struct LocalOOBGroup
    size::Int
    lock::ReentrantLock
    rounds::Dict{Int,LocalOOBRound}
    seq::Vector{Int}  # per-rank allgather sequence number
end

"""
    UCC.LocalOOB(n) :: Vector{LocalOOB}

In-process OOB for `n` ranks living in the same Julia process (one task per
rank). Returns one endpoint per rank. Useful for testing and for driving
several devices from a single process.

When the rank tasks may run on different threads, every UCC library alive in
the process must be thread-safe (`UCC.Library(thread_mode=:multiple)`): UCC
shares global state between library instances, and a single live
`UCC_THREAD_SINGLE` library is enough for concurrent collectives to hang or
crash.

# Examples
```julia
oobs = UCC.LocalOOB(2)
teams = fetch.([Threads.@spawn begin
                    lib = UCC.Library(thread_mode=:multiple)
                    UCC.Team(UCC.Context(oob; lib))
                end for oob in oobs])
```
"""
mutable struct LocalOOB <: AbstractOOB
    const group::LocalOOBGroup
    const rank::Int
end

function LocalOOB(n::Integer)
    let g = LocalOOBGroup(n, ReentrantLock(), Dict{Int,LocalOOBRound}(), zeros(Int, n))
        [LocalOOB(g, r) for r in 0:(n - 1)]
    end
end

oob_rank(oob::LocalOOB) = oob.rank
oob_size(oob::LocalOOB) = oob.group.size

function allgather_start(oob::LocalOOB, src::Ptr{UInt8}, dst::Ptr{UInt8}, nbytes::Int)
    g = oob.group
    @lock g.lock begin
        seq = (g.seq[oob.rank + 1] += 1)
        round = get!(g.rounds, seq) do
            LocalOOBRound([UInt8[] for _ in 1:g.size], 0, 0)
        end
        round.data[oob.rank + 1] = copy(unsafe_wrap(Array, src, nbytes))
        round.arrived += 1
        return (seq, dst, nbytes)
    end
end

function allgather_test(oob::LocalOOB, (seq, dst, nbytes))
    g = oob.group
    @lock g.lock begin
        round = g.rounds[seq]
        round.arrived == g.size || return false
        for (i, d) in enumerate(round.data)
            length(d) == nbytes || error("mismatched OOB allgather sizes")
            GC.@preserve d unsafe_copyto!(dst + (i - 1) * nbytes, pointer(d), nbytes)
        end
        round.collected += 1
        round.collected == g.size && delete!(g.rounds, seq)
        return true
    end
end
