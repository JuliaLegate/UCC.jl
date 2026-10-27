# Run with e.g. `mpiexec -n 3 julia --project=test test/mpi_collectives.jl`

using MPI
using UCC
using Test

MPI.Init()
include("collectives.jl")

ctx = UCC.Context(UCC.MPIOOB(MPI.COMM_WORLD))
team = UCC.Team(ctx)
@test UCC.rank(team) == MPI.Comm_rank(MPI.COMM_WORLD)
@test UCC.size(team) == MPI.Comm_size(MPI.COMM_WORLD)

test_collectives(team, Array)
test_collectives(team, Array; T=Int64)

if get(ENV, "UCC_TEST_CUDA", "false") == "true"
    using CUDA
    test_collectives(team, CuArray)
end

MPI.Comm_rank(MPI.COMM_WORLD) == 0 && println("MPI collectives OK on ", UCC.size(team), " ranks")
UCC.destroy(team)
UCC.destroy(ctx)
MPI.Finalize()
