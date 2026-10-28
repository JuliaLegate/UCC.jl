# UCC.jl

Julia wrapper for [UCC](https://github.com/openucx/ucc), the Unified Collective
Communication library, modeled on [NCCL.jl](https://github.com/JuliaGPU/NCCL.jl).
Collectives work on host `Array`s and on CUDA `CuArray`s.

- `UCC`: `Library`, `Context`, `Team` (≈ communicator) and the collectives.
- `UCC.LibUCC`: low-level bindings to the UCC C API.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaLegate/UCC.jl")
```

UCC itself comes from `UCC_jll`. Requires Julia 1.10 or later.

## Example

```julia
using MPI, UCC
MPI.Init()

ctx  = UCC.Context(UCC.MPIOOB(MPI.COMM_WORLD))  # bootstrap over MPI
team = UCC.Team(ctx)

x = fill(Float64(UCC.rank(team)), 4)            # or a CuArray
UCC.Allreduce!(x, +, team)

UCC.destroy(team)                               # collective: all ranks
UCC.destroy(ctx)
```

Run with `mpiexec -n 4 julia --project script.jl`.

## Collectives

Each collective has a blocking form (returns the receive buffer) and a
nonblocking `I…` form (returns a [`Request`](@ref)):
`Allreduce!`, `Broadcast!`, `Reduce!`, `Allgather!`, `Allgatherv!`, `Alltoall!`,
`Alltoallv!`, `Barrier`, `Fanin`, `Fanout`, `Gather!`, `Gatherv!`,
`ReduceScatter!`, `ReduceScatterv!`, `Scatter!`, `Scatterv!`.

Ranks are 0-based and counts/displacements are in elements. See
[Collectives](api/collectives.md).
