# User-defined datatypes

export GenericDatatype

"""
    UCC.GenericDatatype(T; op=nothing) :: GenericDatatype

Register the `isbits` type `T` with UCC as a contiguous user-defined datatype,
so that arrays of `T` can be used in collectives. If `op` is given (a binary
function `op(a::T, b::T)::T`), the datatype can also be used in reductions
(`Allreduce!`, `Reduce!`, `ReduceScatter!`, …), in which case the `op` argument
of those collectives is ignored. Reductions run on the host, so only host
buffers can be reduced.

Call [`UCC.destroy`](@ref) to unregister the type.

# External links
- `ucc_dt_create_generic`, `ucc_dt_destroy`
"""
mutable struct GenericDatatype{T,F}
    handle::ucc_datatype_t
    op::F
end

const _generic_datatypes = Dict{Type,ucc_datatype_t}()
const _generic_datatype_objs = Dict{Type,GenericDatatype}()  # keep registered types rooted

function _generic_reduce_cb(params::Ptr{ucc_reduce_cb_params_t})::ucc_status_t
    try
        p = unsafe_load(params)
        # UCC 1.8 does not forward `cb_ctx`; fall back to the `context` given to
        # ucc_dt_create_generic (the first field of the internal ucc_dt_generic)
        ctx = p.cb_ctx != C_NULL ? p.cb_ctx : unsafe_load(Ptr{Ptr{Cvoid}}(p.dt))
        dt = unsafe_pointer_to_objref(ctx)::GenericDatatype
        return _generic_reduce(p, dt)
    catch err
        @error "UCC generic datatype reduction failed" exception = (err, catch_backtrace())
        return UCC_ERR_NO_MESSAGE
    end
end

# UCC calls the reduction in two forms:
# - strided: dst = op(src1, src2[0], …, src2[n_vectors-1]), with the src2
#   vectors `stride` bytes apart
# - multi (src2 == NULL): src1 is an array of `n_vectors` source pointers and
#   dst = op(srcs[0], …, srcs[n_vectors-1])
function _generic_reduce(p::ucc_reduce_cb_params_t, dt::GenericDatatype{T}) where {T}
    nv = Int(p.n_vectors)
    if p.src2 == C_NULL
        srcs = [Ptr{T}(unsafe_load(Ptr{Ptr{Cvoid}}(p.src1), v)) for v in 1:nv]
    else
        srcs = [Ptr{T}(p.src1); [Ptr{T}(p.src2 + (v - 1) * p.stride) for v in 1:nv]]
    end
    dst = Ptr{T}(p.dst)
    for i in 1:Int(p.count)
        acc = unsafe_load(srcs[1], i)
        for s in 2:length(srcs)
            acc = dt.op(acc, unsafe_load(srcs[s], i))
        end
        unsafe_store!(dst, acc, i)
    end
    return UCC_OK
end

function GenericDatatype(::Type{T}; op=nothing) where {T}
    isbitstype(T) || throw(ArgumentError("generic datatypes must be isbits, got $T"))
    haskey(_generic_datatypes, T) && throw(ArgumentError("$T is already registered with UCC"))
    dt = GenericDatatype{T,typeof(op)}(0, op)
    flags = UCC_GENERIC_DT_OPS_FLAG_CONTIG
    reduce = LibUCC.ucc_generic_dt_ops_reduce(C_NULL, C_NULL)
    if op !== nothing
        flags |= UCC_GENERIC_DT_OPS_FLAG_REDUCE
        reduce = LibUCC.ucc_generic_dt_ops_reduce(
            @cfunction(_generic_reduce_cb, ucc_status_t, (Ptr{ucc_reduce_cb_params_t},)),
            pointer_from_objref(dt))
    end
    ops = Ref(
        ucc_generic_dt_ops_t(UCC_GENERIC_DT_OPS_FIELD_FLAGS, flags, sizeof(T),
            C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, reduce),
    )
    ref = Ref{ucc_datatype_t}()
    ucc_dt_create_generic(ops, pointer_from_objref(dt), ref)
    dt.handle = ref[]
    _generic_datatypes[T] = dt.handle
    _generic_datatype_objs[T] = dt
    return dt
end

function destroy(dt::GenericDatatype{T}) where {T}
    if dt.handle != 0
        ucc_dt_destroy(dt.handle)
        dt.handle = 0
        delete!(_generic_datatypes, T)
        delete!(_generic_datatype_objs, T)
    end
    return nothing
end
