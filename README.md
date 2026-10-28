# UCC.jl

[![Documentation stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://julialegate.github.io/UCC.jl/stable) [![CI](https://github.com/JuliaLegate/UCC.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/JuliaLegate/UCC.jl/actions/workflows/ci.yml)

Collective communication for Julia on CPUs and GPUs, built on
[UCC](https://github.com/openucx/ucc).

- **One API for `Array`s and `CuArray`s**: UCC picks the transport for each buffer.
- **Blocking and nonblocking** (`I…`) collectives, including persistent and
  CUDA-stream-triggered requests.
- **Bootstraps over MPI** or any transport you plug in.
- **Full UCC 1.8 C API** in `UCC.LibUCC`.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaLegate/UCC.jl")
```

## Example

```julia
using MPI, UCC
MPI.Init()

ctx  = UCC.Context(UCC.MPIOOB(MPI.COMM_WORLD))
team = UCC.Team(ctx)

x = fill(Float64(UCC.rank(team)), 4)  # or a CuArray
UCC.Allreduce!(x, +, team)

UCC.destroy(team)  # collective: all ranks
UCC.destroy(ctx)
```

Run with `mpiexec -n 4 julia --project script.jl`.

See the [documentation](https://julialegate.github.io/UCC.jl/stable) for all
collectives, GPU usage and object lifetimes.
