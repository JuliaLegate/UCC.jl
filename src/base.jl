import UCC_jll: is_available

"""
    UCC.is_available() :: Bool

Is the UCC library available?
"""
is_available

"""
    UCC.version() :: VersionNumber

Get the version of the loaded UCC library.
"""
function version()
    major = Ref{Cuint}()
    minor = Ref{Cuint}()
    release = Ref{Cuint}()
    ucc_get_version(major, minor, release)
    return VersionNumber(major[], minor[], release[])
end

"""
    UCC.version_string() :: String

Get the version string of the loaded UCC library.
"""
version_string() = unsafe_string(ucc_get_version_string())

import .LibUCC: ucc_reduction_op_t, ucc_memory_type_t

ucc_reduction_op_t(::typeof(+)) = UCC_OP_SUM
ucc_reduction_op_t(::typeof(*)) = UCC_OP_PROD
ucc_reduction_op_t(::typeof(max)) = UCC_OP_MAX
ucc_reduction_op_t(::typeof(min)) = UCC_OP_MIN
ucc_reduction_op_t(::typeof(&)) = UCC_OP_BAND
ucc_reduction_op_t(::typeof(|)) = UCC_OP_BOR
ucc_reduction_op_t(::typeof(xor)) = UCC_OP_BXOR
# Handles the case where user directly passed in the ucc_reduction_op_t (eg. `UCC.avg`)
ucc_reduction_op_t(x::ucc_reduction_op_t) = x

"""
    UCC.avg

Perform an average operation, i.e. a sum across all ranks, divided by the number
of ranks.
"""
const avg = UCC_OP_AVG

"""
    UCC.land, UCC.lor, UCC.lxor

Logical and, or and exclusive-or reductions (nonzero is true).
"""
const land = UCC_OP_LAND
const lor = UCC_OP_LOR
const lxor = UCC_OP_LXOR
@doc (@doc land) lor
@doc (@doc land) lxor

_builtin_datatype(::Type{Int8}) = UCC_DT_INT8
_builtin_datatype(::Type{Int16}) = UCC_DT_INT16
_builtin_datatype(::Type{Int32}) = UCC_DT_INT32
_builtin_datatype(::Type{Int64}) = UCC_DT_INT64
_builtin_datatype(::Type{Int128}) = UCC_DT_INT128
_builtin_datatype(::Type{UInt8}) = UCC_DT_UINT8
_builtin_datatype(::Type{UInt16}) = UCC_DT_UINT16
_builtin_datatype(::Type{UInt32}) = UCC_DT_UINT32
_builtin_datatype(::Type{UInt64}) = UCC_DT_UINT64
_builtin_datatype(::Type{UInt128}) = UCC_DT_UINT128
_builtin_datatype(::Type{Float16}) = UCC_DT_FLOAT16
_builtin_datatype(::Type{Float32}) = UCC_DT_FLOAT32
_builtin_datatype(::Type{Float64}) = UCC_DT_FLOAT64
@static if isdefined(Core, :BFloat16)  # Julia 1.11+
    _builtin_datatype(::Type{Core.BFloat16}) = UCC_DT_BFLOAT16
end
_builtin_datatype(::Type{ComplexF32}) = UCC_DT_FLOAT32_COMPLEX
_builtin_datatype(::Type{ComplexF64}) = UCC_DT_FLOAT64_COMPLEX
_builtin_datatype(::Type{Bool}) = UCC_DT_UINT8
function _builtin_datatype(::Type{T}) where {T}
    return throw(
        ArgumentError(
            "no UCC datatype for $T; see `UCC.GenericDatatype` for user-defined types"),
    )
end

"""
    UCC.memory_type(buffer) :: LibUCC.ucc_memory_type_t

The UCC memory type of `buffer`. Host arrays are `UCC_MEMORY_TYPE_HOST`; GPU
array packages add methods through package extensions (e.g. `CuArray`s when
CUDA.jl is loaded). Wrappers such as `SubArray` defer to their parent.
"""
function memory_type(buffer)
    if applicable(parent, buffer)
        p = parent(buffer)
        p === buffer || return memory_type(p)
    end
    return UCC_MEMORY_TYPE_HOST
end
memory_type(::Array) = UCC_MEMORY_TYPE_HOST
memory_type(::Nothing) = UCC_MEMORY_TYPE_HOST

"""
    UCC.synchronize_buffer(buffer)

Called on every buffer before a collective is posted. UCC collectives are
ordered on the host, so device buffers must have finished any pending writes;
the CUDA extension synchronizes the task-local stream here.
"""
function synchronize_buffer(buffer)
    if applicable(parent, buffer)
        p = parent(buffer)
        p === buffer || return synchronize_buffer(p)
    end
    return nothing
end

# raw address of `buffer` as passed to UCC (host or device pointer)
_ptr(::Nothing) = C_NULL
_ptr(buffer) = reinterpret(Ptr{Cvoid}, pointer(buffer))

function _check_contiguous(buffer)
    buffer === nothing && return nothing
    if !(buffer isa DenseArray) &&
        !(
        applicable(strides, buffer) &&
        strides(buffer) == Base.size_to_strides(1, Base.size(buffer)...)
    )
        throw(ArgumentError("UCC buffers must be contiguous; got $(typeof(buffer))"))
    end
end
