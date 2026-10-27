module UCCCUDAExt

using UCC
using UCC.LibUCC
using CUDACore: CUDACore, CuArray, CuStream, is_unified, is_host

function UCC.memory_type(buffer::CuArray)
    is_unified(buffer) && return UCC_MEMORY_TYPE_CUDA_MANAGED
    is_host(buffer) && return UCC_MEMORY_TYPE_HOST
    return UCC_MEMORY_TYPE_CUDA
end

# UCC posts collectives from the host, so pending writes to device buffers on
# the task-local stream must have completed first.
function UCC.synchronize_buffer(buffer::CuArray)
    CUDACore.device!(CUDACore.device(buffer)) do
        CUDACore.synchronize()
    end
end

"""
    UCC.ExecutionEngine(team::UCC.Team, stream::CuStream=CUDACore.stream())

Create a UCC execution engine for the CUDA `stream`, for use with
[`UCC.triggered_post!`](@ref).
"""
function UCC.ExecutionEngine(team::UCC.Team, stream::CuStream=CUDACore.stream())
    handle = Base.unsafe_convert(CUDACore.CUstream, stream)
    UCC.ExecutionEngine(team, UCC_EE_CUDA_STREAM, Ptr{Cvoid}(UInt(handle)),
                        sizeof(CUDACore.CUstream); root=stream)
end

end # module UCCCUDAExt
