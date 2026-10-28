# Execution engines and triggered collectives

"""
    UCC.ExecutionEngine(team::Team, ee_type, ee_context::Ptr, [ee_context_size]) :: ExecutionEngine

Create a UCC execution engine (e.g. a CUDA stream) on `team`. With CUDA.jl
loaded, use `UCC.ExecutionEngine(team, stream::CuStream)`.

Collectives initialized with `post=false` can then be posted in stream order
with [`UCC.triggered_post!`](@ref).

# External links
- `ucc_ee_create`, `ucc_ee_destroy`
"""
mutable struct ExecutionEngine <: UCCObject
    handle::ucc_ee_h
    team::Team
    context::Any  # keeps the underlying resource (e.g. stream) alive
    @atomic refs::Int
    @atomic destroyed::Bool
end

function ExecutionEngine(team::Team, ee_type::ucc_ee_type_t, ee_context::Ptr,
                         ee_context_size::Integer=0; root=nothing)
    _alive(team)
    params = Ref(ucc_ee_params_t(ee_type, ee_context, ee_context_size))
    ref = Ref{ucc_ee_h}(C_NULL)
    ucc_ee_create(team, params, ref)
    ee = ExecutionEngine(ref[], _retain(team), root, 0, false)
    return finalizer(_defer, ee)
end

Base.unsafe_convert(::Type{ucc_ee_h}, ee::ExecutionEngine) = ee.handle

function _free(ee::ExecutionEngine)
    if ee.handle != C_NULL
        ucc_ee_destroy(ee)
        ee.handle = C_NULL
        _release(ee.team)
    end
end

"""
    UCC.triggered_post!(ee::ExecutionEngine, req::Request;
                        ev_type=UCC_EVENT_COMPUTE_COMPLETE) :: Request

Post `req` (initialized with `post=false`) so that it starts once the work
already enqueued on `ee` has completed. Complete it with `wait(req)` as usual.

# External links
- `ucc_collective_triggered_post`
"""
function triggered_post!(ee::ExecutionEngine, req::Request;
                         ev_type::ucc_event_type_t=UCC_EVENT_COMPUTE_COMPLETE)
    req.handle == C_NULL && throw(ArgumentError("request has been freed"))
    _alive(ee)
    ev = Ref(ucc_ev_t(ev_type, C_NULL, 0, req.handle))
    ucc_collective_triggered_post(ee, ev)
    if req.ee !== ee
        req.ee === nothing || _release(req.ee)
        req.ee = _retain(ee)
    end
    req.posted = true
    return req
end

"""
    UCC.get_event(ee::ExecutionEngine) :: Union{Ptr{ucc_ev_t},Nothing}

Retrieve the next pending event of `ee` (or `nothing`). Acknowledge it with
[`UCC.ack_event!`](@ref).

# External links
- `ucc_ee_get_event`
"""
function get_event(ee::ExecutionEngine)
    ev = Ref{Ptr{ucc_ev_t}}(C_NULL)
    st = ucc_ee_get_event(ee, ev)
    st == UCC_ERR_NOT_FOUND && return nothing
    LibUCC.check(st)
    return ev[]
end

"""
    UCC.ack_event!(ee::ExecutionEngine, ev::Ptr{ucc_ev_t})

Acknowledge an event obtained with [`UCC.get_event`](@ref).
"""
ack_event!(ee::ExecutionEngine, ev::Ptr{ucc_ev_t}) = (ucc_ee_ack_event(ee, ev); nothing)

"""
    UCC.set_event!(ee::ExecutionEngine, ev::Ref{ucc_ev_t})

Notify `ee` of an event.
"""
set_event!(ee::ExecutionEngine, ev) = (ucc_ee_set_event(ee, ev); nothing)

"""
    wait(ee::ExecutionEngine, ev)

Block until the event `ev` is triggered on `ee`.
"""
Base.wait(ee::ExecutionEngine, ev) = (ucc_ee_wait(ee, ev); nothing)
