# Library

export Library

# UCC objects form a hierarchy (library ⊃ context ⊃ team) and a parent must
# outlive its children. GC finalizers run in no particular order, so each
# object counts its live children and only frees its handle once it has been
# destroyed *and* has no children left.
abstract type UCCObject end

function _retain(x::UCCObject)
    @atomic x.refs += 1
    return x
end

function _release(x::UCCObject)
    if (@atomic x.refs -= 1) == 0 && @atomic(x.destroyed)
        _free(x)
    end
    return
end

"""
    UCC.destroy(x)

Release the UCC resources held by `x` (a [`Library`](@ref), [`Context`](@ref),
[`Team`](@ref), …). Resources of a parent object are only released once all of
its children have been destroyed as well.
"""
function destroy(x::UCCObject)
    _, ok = @atomicreplace x.destroyed false => true
    ok && @atomic(x.refs) == 0 && _free(x)
    return nothing
end

# Deferred destruction
#
# GC finalizers run on whichever thread triggers a collection (including the
# interactive thread), at any allocation point. UCC is normally initialized as
# UCC_THREAD_SINGLE, and even a thread-safe library must not be finalized while
# another thread is inside UCC, so finalizers never call into UCC. They push the
# object onto a lock-free list (taking a lock could deadlock if the GC ran while
# this thread held it), which is only drained by an explicit
# `UCC.free_deferred()` at a point the application knows to be quiescent.

struct DeferredNode
    obj::Any
    next::Union{Nothing,DeferredNode}
end

mutable struct DeferredList
    @atomic head::Union{Nothing,DeferredNode}
end

const _deferred = DeferredList(nothing)

# finalizer: enqueue `x` for destruction (resurrecting it until then)
function _defer(x)
    head = @atomic _deferred.head
    while true
        head, ok = @atomicreplace _deferred.head head => DeferredNode(x, head)
        ok && return
    end
end

"""
    UCC.free_deferred()

Release the UCC resources of objects that were garbage collected without an
explicit [`UCC.destroy`](@ref). Finalizers never call into UCC themselves (they
may run on any thread, concurrently with UCC calls elsewhere), so call this
from the task that drives UCC, while no other thread is using UCC.
"""
function free_deferred()
    @atomic(_deferred.head) === nothing && return
    node = @atomicswap _deferred.head = nothing
    while node !== nothing
        try
            destroy(node.obj)
        catch err
            @error "UCC: error while freeing $(typeof(node.obj))" exception = (err, catch_backtrace())
        end
        node = node.next
    end
end

"""
    UCC.LibraryConfig(; env_prefix=nothing, filename=nothing) :: LibraryConfig

Read the UCC library configuration from the environment (variables prefixed by
`UCC_` or `<env_prefix>_UCC_`) and, optionally, a configuration file. Values can
be changed with `config[name] = value`.

# External links
- `ucc_lib_config_read`, `ucc_lib_config_modify`
"""
mutable struct LibraryConfig
    handle::ucc_lib_config_h
end

function LibraryConfig(; env_prefix::Union{Nothing,AbstractString}=nothing,
                       filename::Union{Nothing,AbstractString}=nothing)
    ref = Ref{ucc_lib_config_h}(C_NULL)
    ucc_lib_config_read(_cstr(env_prefix), _cstr(filename), ref)
    return finalizer(_defer, LibraryConfig(ref[]))
end

Base.unsafe_convert(::Type{ucc_lib_config_h}, c::LibraryConfig) = c.handle

destroy(config::LibraryConfig) = release(config)

function release(config::LibraryConfig)
    if config.handle != C_NULL
        ucc_lib_config_release(config)
        config.handle = C_NULL
    end
    return nothing
end

function Base.setindex!(config::LibraryConfig, value, name::AbstractString)
    ucc_lib_config_modify(config, name, string(value))
    return config
end

Base.show(io::IO, ::MIME"text/plain", config::LibraryConfig) =
    print(io, _config_string((s, t, f) -> ucc_lib_config_print(config, s, t, f),
                             "UCC library configuration"))

_cstr(::Nothing) = Ptr{Cchar}(C_NULL)
_cstr(s::AbstractString) = s

# capture the output of a `ucc_*_config_print(…, FILE *stream, …)` call
function _config_string(print_fn, title;
                        flags=UCC_CONFIG_PRINT_CONFIG | UCC_CONFIG_PRINT_HEADER)
    mktemp() do path, io
        close(io)
        stream = ccall(:fopen, Ptr{Cvoid}, (Cstring, Cstring), path, "w")
        stream == C_NULL && Base.systemerror("fopen")
        try
            print_fn(stream, title, flags)
        finally
            ccall(:fclose, Cint, (Ptr{Cvoid},), stream)
        end
        read(path, String)
    end
end

Base.:(|)(a::ucc_config_print_flags_t, b::ucc_config_print_flags_t) =
    ucc_config_print_flags_t(UInt32(a) | UInt32(b))

"""
    UCC.Library(; thread_mode, coll_types, reduction_types, sync_type,
                config::LibraryConfig) :: Library

Initialize a UCC library instance. Every keyword is optional and only passed to
UCC when given:

- `thread_mode`: `:single`, `:funneled` or `:multiple` (or the corresponding
  `UCC_THREAD_*` value)
- `coll_types`: bitmask of `UCC_COLL_TYPE_*` the application will use
- `reduction_types`: bitmask of `1 << UCC_OP_*` the application will use
- `sync_type`: `UCC_NO_SYNC_COLLECTIVES` or `UCC_SYNC_COLLECTIVES`

# External links
- `ucc_init`, `ucc_lib_get_attr`, `ucc_finalize`
"""
mutable struct Library <: UCCObject
    handle::ucc_lib_h
    @atomic refs::Int
    @atomic destroyed::Bool
end

_thread_mode(mode::ucc_thread_mode_t) = mode
function _thread_mode(mode::Symbol)
    mode === :single && return UCC_THREAD_SINGLE
    mode === :funneled && return UCC_THREAD_FUNNELED
    mode === :multiple && return UCC_THREAD_MULTIPLE
    throw(ArgumentError("unknown thread mode $(repr(mode))"))
end

function Library(; thread_mode::Union{Nothing,Symbol,ucc_thread_mode_t}=nothing,
                 coll_types::Union{Nothing,Integer}=nothing,
                 reduction_types::Union{Nothing,Integer}=nothing,
                 sync_type::Union{Nothing,ucc_coll_sync_type_t}=nothing,
                 config::Union{Nothing,LibraryConfig}=nothing)
    mask = UInt64(0)
    thread_mode === nothing || (mask |= UCC_LIB_PARAM_FIELD_THREAD_MODE)
    coll_types === nothing || (mask |= UCC_LIB_PARAM_FIELD_COLL_TYPES)
    reduction_types === nothing || (mask |= UCC_LIB_PARAM_FIELD_REDUCTION_TYPES)
    sync_type === nothing || (mask |= UCC_LIB_PARAM_FIELD_SYNC_TYPE)
    params = Ref(ucc_lib_params_t(mask,
                                  _thread_mode(something(thread_mode, :single)),
                                  something(coll_types, 0),
                                  something(reduction_types, 0),
                                  something(sync_type, UCC_NO_SYNC_COLLECTIVES)))
    # UCC requires a configuration; read the default (environment) one if needed
    cfg = config === nothing ? LibraryConfig() : config
    ref = Ref{ucc_lib_h}(C_NULL)
    try
        ucc_init(params, cfg, ref)
    finally
        config === nothing && release(cfg)
    end
    return finalizer(_defer, Library(ref[], 0, false))
end

Base.unsafe_convert(::Type{ucc_lib_h}, lib::Library) = lib.handle

function _free(lib::Library)
    if lib.handle != C_NULL
        ucc_finalize(lib)
        lib.handle = C_NULL
    end
end

"""
    UCC.attributes(lib::Library) :: LibUCC.ucc_lib_attr_t

Query the attributes of `lib` selected by `mask` (by default the thread mode and
supported collective types; UCC 1.8 does not report reduction or sync types).
"""
function attributes(lib::Library;
                    mask=UCC_LIB_ATTR_FIELD_THREAD_MODE | UCC_LIB_ATTR_FIELD_COLL_TYPES)
    attr = Ref(ucc_lib_attr_t(mask, UCC_THREAD_SINGLE, 0, 0, UCC_NO_SYNC_COLLECTIVES))
    ucc_lib_get_attr(lib, attr)
    return attr[]
end

"""
    UCC.thread_mode(lib::Library) :: LibUCC.ucc_thread_mode_t

The thread mode provided by `lib`.
"""
thread_mode(lib::Library) = attributes(lib; mask=UCC_LIB_ATTR_FIELD_THREAD_MODE).thread_mode
