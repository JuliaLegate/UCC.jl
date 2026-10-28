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
