# Collective requests

export Request

"""
    UCC.Request

A collective operation returned by the nonblocking collectives (`UCC.Iallreduce!`,
…). Complete it with `wait(req)` (which returns the receive buffer) or poll it
with [`UCC.test`](@ref).

Persistent requests (created with `persistent=true`) can be re-posted with
[`UCC.post!`](@ref) after completing, and must be released with
[`UCC.free`](@ref).

# External links
- `ucc_collective_post`, `ucc_collective_test`, `ucc_collective_finalize`
"""
mutable struct Request
    handle::ucc_coll_req_h
    team::Team
    buffers::Any       # rooted until the request is freed
    result::Any        # returned by `wait`
    persistent::Bool
    posted::Bool
end

Base.unsafe_convert(::Type{ucc_coll_req_h}, req::Request) = req.handle

"""
    UCC.status(req::Request) :: LibUCC.ucc_status_t

The raw UCC status of `req` (`UCC_OK`, `UCC_INPROGRESS`,
`UCC_OPERATION_INITIALIZED` or an error code).
"""
status(req::Request) = req.handle == C_NULL ? UCC_OK : ucc_collective_test(req.handle)

"""
    UCC.post!(req::Request) :: Request

Post an initialized (`post=false`) or completed persistent request.

# External links
- `ucc_collective_post`
"""
function post!(req::Request)
    req.handle == C_NULL && throw(ArgumentError("request has been freed"))
    foreach(synchronize_buffer, req.buffers)
    ucc_collective_post(req)
    req.posted = true
    return req
end

"""
    UCC.test(req::Request) :: Bool

Progress the team's context once and return whether `req` has completed. A
completed non-persistent request is freed. Throws a `UCCError` if the
collective failed.
"""
function test(req::Request)
    req.handle == C_NULL && return true
    req.posted || throw(ArgumentError("request has not been posted"))
    st = ucc_collective_test(req.handle)
    if st == UCC_INPROGRESS
        ucc_context_progress(req.team.ctx)
        st = ucc_collective_test(req.handle)
    end
    st == UCC_INPROGRESS && return false
    if Int32(st) < 0
        free(req)
        throw(UCCError(st))
    end
    req.persistent || free(req)
    return true
end

"""
    wait(req::Request)

Block (yielding to other tasks) until `req` completes, and return its receive
buffer (or `nothing` for barrier-like collectives).
"""
function Base.wait(req::Request)
    while !test(req)
        yield()
    end
    return req.result
end

"""
    UCC.free(req::Request)

Release the resources of `req`. Non-persistent requests are freed automatically
when they complete.

# External links
- `ucc_collective_finalize`
"""
function free(req::Request)
    if req.handle != C_NULL
        ucc_collective_finalize(req)
        req.handle = C_NULL
        req.buffers = ()
    end
    return nothing
end

const _no_buffer = ucc_coll_buffer_info_u()
const _no_active_set = LibUCC.ucc_coll_args_active_set(0, 0, 0)

# Initialize (and by default post) a collective on `team`.
function _collective(team::Team, coll_type::ucc_coll_type_t;
                     src::ucc_coll_buffer_info_u=_no_buffer,
                     dst::ucc_coll_buffer_info_u=_no_buffer,
                     op::ucc_reduction_op_t=UCC_OP_SUM,
                     root::Integer=0,
                     flags::UInt64=UInt64(0),
                     buffers::Tuple=(),
                     result=nothing,
                     post::Bool=true,
                     persistent::Bool=false,
                     timeout::Union{Nothing,Real}=nothing,
                     tag::Union{Nothing,Integer}=nothing)
    team.handle == C_NULL && throw(ArgumentError("team has been destroyed"))
    0 <= root < team.size || throw(ArgumentError("root must be in [0, $(team.size))"))
    mask = UCC_COLL_ARGS_FIELD_FLAGS
    flags |= UCC_COLL_ARGS_FLAG_COUNT_64BIT | UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT
    persistent && (flags |= UCC_COLL_ARGS_FLAG_PERSISTENT)
    timeout === nothing || (flags |= UCC_COLL_ARGS_FLAG_TIMEOUT)
    tag === nothing || (mask |= UCC_COLL_ARGS_FIELD_TAG)
    args = Ref(ucc_coll_args_t(mask, coll_type, src, dst, op, flags, root,
                               UCC_ERR_TYPE_LOCAL, something(tag, 0), C_NULL,
                               ucc_coll_callback_t(C_NULL, C_NULL),
                               something(timeout, 0.0), _no_active_set, C_NULL, C_NULL))
    ref = Ref{ucc_coll_req_h}(C_NULL)
    GC.@preserve buffers ucc_collective_init(args, ref, team)
    req = Request(ref[], team, buffers, result, persistent, false)
    if post
        try
            post!(req)
        catch
            free(req)
            rethrow()
        end
    end
    return req
end

# blocking wrapper: wait for (and release) the request
function _blocking(req::Request)
    result = wait(req)
    req.persistent && free(req)
    return result
end
