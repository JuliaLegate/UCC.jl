# Memory mapping (for one-sided collectives)

"""
    UCC.MemoryHandle(ctx::Context, buffers...; mode=UCC_MEM_MAP_MODE_EXPORT) :: MemoryHandle

Map (register) `buffers` with `ctx` for use in one-sided collectives. With
`UCC_MEM_MAP_MODE_EXPORT`, the serialized handle (`Vector{UInt8}(memh)`) can be
exchanged with other ranks, which then import it with
`UCC.MemoryHandle(ctx, bytes; mode=UCC_MEM_MAP_MODE_IMPORT)`.

Release with [`UCC.destroy`](@ref).

# External links
- `ucc_mem_map`, `ucc_mem_unmap`
"""
mutable struct MemoryHandle
    handle::ucc_mem_map_mem_h
    size::Int
    ctx::Context
    buffers::Any
end

function MemoryHandle(ctx::Context, buffers...; mode::ucc_mem_map_mode_t=UCC_MEM_MAP_MODE_EXPORT)
    return _map_buffers(ctx, buffers; mode)
end

function _map_buffers(ctx::Context, buffers::Tuple; mode::ucc_mem_map_mode_t)
    segments = [ucc_mem_map_t(_ptr(b), sizeof(b)) for b in buffers]
    params = Ref(ucc_mem_map_params_t(pointer(segments), length(segments)))
    memh_size = Ref{Csize_t}(0)
    memh = Ref{ucc_mem_map_mem_h}(C_NULL)
    GC.@preserve buffers segments ucc_mem_map(_alive(ctx), mode, params, memh_size, memh)
    return finalizer(_defer, MemoryHandle(memh[], Int(memh_size[]), _retain(ctx), buffers))
end

"""
    UCC.MemoryHandle(ctx::Context, bytes::Vector{UInt8}; mode=UCC_MEM_MAP_MODE_IMPORT)

Import a memory handle that was exported (and serialized) by another rank.
"""
function MemoryHandle(ctx::Context, bytes::Vector{UInt8};
    mode::ucc_mem_map_mode_t=UCC_MEM_MAP_MODE_EXPORT)
    mode == UCC_MEM_MAP_MODE_EXPORT && return _map_buffers(ctx, (bytes,); mode)
    # on import UCC reads the user's handle from *memh and updates it in place
    buf = Libc.malloc(length(bytes))
    unsafe_copyto!(Ptr{UInt8}(buf), pointer(bytes), length(bytes))
    memh = Ref{ucc_mem_map_mem_h}(buf)
    memh_size = Ref{Csize_t}(length(bytes))
    params = Ref(ucc_mem_map_params_t(C_NULL, 0))
    try
        ucc_mem_map(_alive(ctx), mode, params, memh_size, memh)
    catch
        Libc.free(buf)
        rethrow()
    end
    return finalizer(_defer, MemoryHandle(memh[], Int(memh_size[]), _retain(ctx), bytes))
end

function Base.Vector{UInt8}(memh::MemoryHandle)
    return copy(unsafe_wrap(Array, Ptr{UInt8}(memh.handle), memh.size))
end

function destroy(memh::MemoryHandle)
    if memh.handle != C_NULL
        ucc_mem_unmap(Ref(memh.handle))
        memh.handle = C_NULL
        memh.buffers = nothing
        _release(memh.ctx)
    end
    return nothing
end
