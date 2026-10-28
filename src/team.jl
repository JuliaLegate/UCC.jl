# Team

export Team

"""
    UCC.Team(ctx::Context, [oob::AbstractOOB=ctx.oob];
             ordering, outstanding_colls, sync_type, id) :: Team

Create a team: the UCC analogue of a communicator. All ranks of `oob` must
call `Team` collectively; the calling process becomes rank `oob_rank(oob)`.
Pass a different `oob` (over a subset of the context's ranks) to create a
sub-team.

Optional keywords (only passed to UCC when given):

- `ordering`: a `UCC_COLLECTIVE_*` value of `LibUCC.ucc_post_ordering_t`
- `outstanding_colls`: max number of outstanding collectives
- `sync_type`: `UCC_NO_SYNC_COLLECTIVES` or `UCC_SYNC_COLLECTIVES`
- `id`: unique team id

Teams are not freed by the garbage collector because destruction is a
collective operation: call [`UCC.destroy`](@ref) on all ranks when done.

# Examples
```julia
ctx  = UCC.Context(UCC.MPIOOB(MPI.COMM_WORLD))   # requires `using MPI`
team = UCC.Team(ctx)
UCC.Allreduce!(x, +, team)
```

# External links
- `ucc_team_create_post`, `ucc_team_create_test`, `ucc_team_destroy`
"""
mutable struct Team <: UCCObject
    handle::ucc_team_h
    ctx::Context
    oob::Union{Nothing,AbstractOOB}
    rank::Int
    size::Int
    @atomic refs::Int
    @atomic destroyed::Bool
end

function Team(ctx::Context, oob::AbstractOOB=_context_oob(ctx);
    ordering::Union{Nothing,ucc_post_ordering_t}=nothing,
    outstanding_colls::Union{Nothing,Integer}=nothing,
    sync_type::Union{Nothing,ucc_coll_sync_type_t}=nothing,
    id::Union{Nothing,Integer}=nothing)
    mask = UCC_TEAM_PARAM_FIELD_OOB | UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_EP_RANGE
    ordering === nothing || (mask |= UCC_TEAM_PARAM_FIELD_ORDERING)
    outstanding_colls === nothing || (mask |= UCC_TEAM_PARAM_FIELD_OUTSTANDING_COLLS)
    sync_type === nothing || (mask |= UCC_TEAM_PARAM_FIELD_SYNC_TYPE)
    id === nothing || (mask |= UCC_TEAM_PARAM_FIELD_ID)
    params = Ref(
        ucc_team_params_t(
            mask, 0,
            something(ordering, UCC_COLLECTIVE_POST_ORDERED),
            something(outstanding_colls, 0),
            oob_rank(oob),
            C_NULL,
            UCC_COLLECTIVE_EP_RANGE_CONTIG,
            oob_size(oob),
            something(sync_type, UCC_NO_SYNC_COLLECTIVES),
            _oob_coll(oob),
            ucc_team_p2p_conn_t(C_NULL, C_NULL, C_NULL, C_NULL, C_NULL),
            ucc_mem_map_params_t(C_NULL, 0),
            LibUCC.ucc_ep_map_t(UCC_EP_MAP_FULL, 0),
            something(id, 0)),
    )
    ref = Ref{ucc_team_h}(C_NULL)
    GC.@preserve oob begin
        ucc_team_create_post(Ref(_alive(ctx).handle), 1, params, ref)
        _team_create_wait(ctx, ref[])
    end
    return _team(ref[], ctx, oob)
end

function _context_oob(ctx::Context)
    ctx.oob === nothing &&
        throw(ArgumentError("context was created without an OOB; pass one to `Team`"))
    return ctx.oob
end

function _team_create_wait(ctx::Context, handle::ucc_team_h)
    while ucc_team_create_test(handle) == UCC_INPROGRESS
        ucc_context_progress(ctx)
        yield()
    end
end

function _team(handle::ucc_team_h, ctx::Context, oob)
    team = Team(handle, _retain(ctx), oob, -1, -1, 0, false)
    attr = attributes(team; mask=UCC_TEAM_ATTR_FIELD_EP | UCC_TEAM_ATTR_FIELD_SIZE)
    team.rank = Int(attr.ep)
    team.size = Int(attr.size)
    return team
end

Base.unsafe_convert(::Type{ucc_team_h}, team::Team) = team.handle

function _free(team::Team)
    if team.handle != C_NULL
        while ucc_team_destroy(team) == UCC_INPROGRESS
            ucc_context_progress(team.ctx)
            yield()
        end
        team.handle = C_NULL
        _release(team.ctx)
    end
end

"""
    UCC.attributes(team::Team) :: LibUCC.ucc_team_attr_t

Query the attributes of `team`.
"""
function attributes(team::Team;
    mask=UCC_TEAM_ATTR_FIELD_POST_ORDERING | UCC_TEAM_ATTR_FIELD_EP |
         UCC_TEAM_ATTR_FIELD_SYNC_TYPE | UCC_TEAM_ATTR_FIELD_SIZE)
    attr = Ref(
        ucc_team_attr_t(mask, UCC_COLLECTIVE_POST_ORDERED, 0, 0,
            UCC_COLLECTIVE_EP_RANGE_CONTIG, UCC_NO_SYNC_COLLECTIVES,
            ucc_mem_map_params_t(C_NULL, 0), 0, C_NULL),
    )
    ucc_team_get_attr(team, attr)
    return attr[]
end

"""
    UCC.rank(team::Team) :: Int

The 0-based index of the calling process in `team`.
"""
rank(team::Team) = team.rank

"""
    UCC.size(team::Team) :: Int

The number of ranks in `team`.
"""
size(team::Team) = team.size

"""
    UCC.context(team::Team) :: Context

The context `team` was created on.
"""
context(team::Team) = team.ctx

function Base.show(io::IO, team::Team)
    return print(io, "UCC.Team(rank=", team.rank, ", size=", team.size,
        team.handle == C_NULL ? ", destroyed" : "", ")")
end
