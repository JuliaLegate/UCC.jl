# UCC.jl

Julia wrapper for [UCC](https://github.com/openucx/ucc) (from `UCC_jll`), modeled on
[NCCL.jl](https://github.com/JuliaGPU/NCCL.jl).

- `UCC.LibUCC`: raw bindings for the full UCC 1.8 C API.
- `UCC`: `Library`, `Context`, `Team` (≈ communicator) and the collectives.

## Example

```julia
using MPI, UCC
MPI.Init()

ctx  = UCC.Context(UCC.MPIOOB(MPI.COMM_WORLD))  # OOB bootstrap over MPI
team = UCC.Team(ctx)

x = fill(Float64(UCC.rank(team)), 4)            # or a CuArray
UCC.Allreduce!(x, +, team)

UCC.destroy(team)                               # collective: all ranks
UCC.destroy(ctx)
```

## Collectives

Blocking (returns the receive buffer) and nonblocking `I…` (returns a `Request`):
`Allreduce!`, `Broadcast!`, `Reduce!`, `Allgather!`, `Allgatherv!`, `Alltoall!`,
`Alltoallv!`, `Barrier`, `Fanin`, `Fanout`, `Gather!`, `Gatherv!`,
`ReduceScatter!`, `ReduceScatterv!`, `Scatter!`, `Scatterv!`.

Ranks are 0-based; counts/displacements are in elements. Nonblocking calls
accept `persistent`, `post=false` (for `UCC.triggered_post!` on a CUDA-stream
`ExecutionEngine`), `timeout` and `tag`. Custom element types:
`UCC.GenericDatatype(T; op)`.

## Notes

- **OOB**: implement `UCC.AbstractOOB` (`oob_rank`, `oob_size`, `allgather`) for
  other transports. `UCC.LocalOOB(n)` runs `n` ranks as tasks in one process;
  if they can run on different threads, *every* library in the process must be
  `UCC.Library(thread_mode=:multiple)`.
- **GC**: finalizers never call into UCC, since they may run on any thread
  (including the interactive one) concurrently with UCC calls. Collected objects
  are queued until you call `UCC.free_deferred()` at a quiescent point. Prefer
  `UCC.destroy` (teams must be destroyed explicitly, on all ranks).
- **Tests**: `julia --project=test -t 4 test/runtests.jl` (in-process ranks,
  plus `mpiexec` with CUDA buffers when a GPU is available). The tests set
  `UCX_TLS=^tcp`: UCX's TCP transport hangs in a netlink query on some machines.
