# Context

export Context

"""
    UCC.ContextConfig(lib::Library; filename=nothing) :: ContextConfig

Read the UCC context configuration for `lib` from the environment and,
optionally, a configuration file. Values can be changed with
`config[name] = value` (UCC core) or `config[component, name] = value`
(a specific component, e.g. `config["tl/ucp", "TUNE"] = "..."`).

# External links
- `ucc_context_config_read`, `ucc_context_config_modify`
"""
mutable struct ContextConfig
    handle::ucc_context_config_h
    lib::Library
end

function ContextConfig(lib::Library; filename::Union{Nothing,AbstractString}=nothing)
    ref = Ref{ucc_context_config_h}(C_NULL)
    ucc_context_config_read(_alive(lib), _cstr(filename), ref)
    return finalizer(_defer, ContextConfig(ref[], _retain(lib)))
end

Base.unsafe_convert(::Type{ucc_context_config_h}, c::ContextConfig) = c.handle

destroy(config::ContextConfig) = release(config)

function release(config::ContextConfig)
    if config.handle != C_NULL
        ucc_context_config_release(config)
        config.handle = C_NULL
        _release(config.lib)
    end
    return nothing
end

function Base.setindex!(config::ContextConfig, value, name::AbstractString)
    ucc_context_config_modify(config, _cstr(nothing), name, string(value))
    return config
end

function Base.setindex!(config::ContextConfig, value, component::AbstractString,
    name::AbstractString)
    ucc_context_config_modify(config, component, name, string(value))
    return config
end

function Base.show(io::IO, ::MIME"text/plain", config::ContextConfig)
    return print(
        io,
        _config_string((s, t, f) -> ucc_context_config_print(config, s, t, f),
            "UCC context configuration"),
    )
end

"""
    UCC.Context([oob::AbstractOOB]; lib, type, sync_type, id,
                config::ContextConfig) :: Context

Create a UCC context. With an `oob`, the context is wired up with all OOB
participants (this blocks until every rank has called `Context`); a [`Team`](@ref)
can then be created over the same ranks with `Team(ctx)`. Without `lib`, a
default [`Library`](@ref) is created and destroyed together with the context.

Optional keywords (only passed to UCC when given):

- `type`: `UCC_CONTEXT_EXCLUSIVE` or `UCC_CONTEXT_SHARED`
- `sync_type`: `UCC_NO_SYNC_COLLECTIVES` or `UCC_SYNC_COLLECTIVES`
- `id`: unique context id

# External links
- `ucc_context_create`, `ucc_context_destroy`
"""
mutable struct Context <: UCCObject
    handle::ucc_context_h
    lib::Library
    owns_lib::Bool  # `lib` was created for this context and is destroyed with it
    oob::Union{Nothing,AbstractOOB}
    @atomic refs::Int
    @atomic destroyed::Bool
end

function Context(oob::Union{Nothing,AbstractOOB}=nothing;
    lib::Union{Nothing,Library}=nothing,
    type::Union{Nothing,ucc_context_type_t}=nothing,
    sync_type::Union{Nothing,ucc_coll_sync_type_t}=nothing,
    id::Union{Nothing,Integer}=nothing,
    config::Union{Nothing,ContextConfig}=nothing)
    mask = UInt64(0)
    type === nothing || (mask |= UCC_CONTEXT_PARAM_FIELD_TYPE)
    sync_type === nothing || (mask |= UCC_CONTEXT_PARAM_FIELD_SYNC_TYPE)
    oob === nothing || (mask |= UCC_CONTEXT_PARAM_FIELD_OOB)
    id === nothing || (mask |= UCC_CONTEXT_PARAM_FIELD_ID)
    oob_coll = if oob === nothing
        LibUCC.ucc_oob_coll_t(C_NULL, C_NULL, C_NULL, C_NULL, 0, 0)
    else
        _oob_coll(oob)
    end
    params = Ref(
        ucc_context_params_t(mask,
            something(type, UCC_CONTEXT_EXCLUSIVE),
            something(sync_type, UCC_NO_SYNC_COLLECTIVES),
            oob_coll,
            something(id, 0),
            ucc_mem_map_params_t(C_NULL, 0)),
    )
    owns_lib = lib === nothing
    owns_lib && (lib = Library())
    # UCC requires a configuration; read the default (environment) one if needed
    cfg = config === nothing ? ContextConfig(lib) : config
    ref = Ref{ucc_context_h}(C_NULL)
    try
        GC.@preserve oob ucc_context_create(_alive(lib), params, cfg, ref)
    finally
        config === nothing && release(cfg)
    end
    ctx = Context(ref[], _retain(lib), owns_lib, oob, 0, false)
    return finalizer(_defer, ctx)
end

Base.unsafe_convert(::Type{ucc_context_h}, ctx::Context) = ctx.handle

function _free(ctx::Context)
    if ctx.handle != C_NULL
        ucc_context_destroy(ctx)
        ctx.handle = C_NULL
        _release(ctx.lib)
        ctx.owns_lib && destroy(ctx.lib)
    end
end

"""
    UCC.progress(ctx::Context)

Progress outstanding operations on `ctx`. Blocking calls in this package
progress the context themselves; this is only needed when polling nonblocking
[`Request`](@ref)s manually.

# External links
- `ucc_context_progress`
"""
function progress(ctx::Context)
    ucc_context_progress(ctx)
    return nothing
end

"""
    UCC.attributes(ctx::Context) :: LibUCC.ucc_context_attr_t

Query the type, sync type, address and global work buffer size of `ctx`.
"""
function attributes(ctx::Context;
    mask=UCC_CONTEXT_ATTR_FIELD_TYPE | UCC_CONTEXT_ATTR_FIELD_SYNC_TYPE |
         UCC_CONTEXT_ATTR_FIELD_CTX_ADDR | UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN |
         UCC_CONTEXT_ATTR_FIELD_WORK_BUFFER_SIZE)
    attr = Ref(
        ucc_context_attr_t(mask, UCC_CONTEXT_EXCLUSIVE, UCC_NO_SYNC_COLLECTIVES,
            C_NULL, 0, 0),
    )
    ucc_context_get_attr(ctx, attr)
    return attr[]
end

"""
    UCC.address(ctx::Context) :: Vector{UInt8}

The serialized address of `ctx`, which can be exchanged to create teams without
an OOB.
"""
function address(ctx::Context)
    attr = attributes(
        ctx; mask=UCC_CONTEXT_ATTR_FIELD_CTX_ADDR |
                  UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN
    )
    return copy(unsafe_wrap(Array, Ptr{UInt8}(attr.ctx_addr), attr.ctx_addr_len))
end

"""
    UCC.work_buffer_size(ctx::Context) :: Int

Size in bytes of the global work buffer that one-sided collectives require.
"""
function work_buffer_size(ctx::Context)
    return Int(
        attributes(ctx; mask=UCC_CONTEXT_ATTR_FIELD_WORK_BUFFER_SIZE).global_work_buffer_size
    )
end
