# GPU Buffers

With CUDA.jl loaded, collectives accept `CuArray`s. UCC picks a transport from
each buffer's [`UCC.memory_type`](@ref) (device, managed or host).

```julia
using CUDA, MPI, UCC
MPI.Init()
CUDA.device!(MPI.Comm_rank(MPI.COMM_WORLD) % length(CUDA.devices()))  # before the context

ctx  = UCC.Context(UCC.MPIOOB(MPI.COMM_WORLD))
team = UCC.Team(ctx)
x = CUDA.ones(Float32, 1024)
UCC.Allreduce!(x, +, team)
```

- Select the GPU before creating the context; UCC uses the current device.
- Before posting, UCC.jl waits for pending work on the buffer's task-local stream
  ([`UCC.synchronize_buffer`](@ref)).
- To post in stream order instead, create the request with `post=false` and
  post it with [`UCC.triggered_post!`](@ref) on an
  [`ExecutionEngine`](@ref UCC.ExecutionEngine) for a `CuStream`.
- Some reductions only run on the GPU, e.g. `Float16`.

::: warning CUDA.jl memory pool
UCC's CUDA transport (`tl_cuda`) shares buffers between ranks over CUDA IPC,
which does not support CUDA.jl's default memory pool. Collectives it handles
then fail with `UCC_ERR_INVALID_PARAM`. Either disable the pool with
`JULIA_CUDA_MEMORY_POOL=none`, or disable the transport with
`UCC_CL_BASIC_TLS=^cuda`.
:::
