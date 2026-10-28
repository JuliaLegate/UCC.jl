# Run with e.g. `mpiexec -n 3 julia --project=test test/mpi_collectives.jl`

using MPI
using UCC
using Test

MPI.Init()
include("collectives.jl")

const TEST_CUDA = get(ENV, "UCC_TEST_CUDA", "false") == "true"
if TEST_CUDA
    using CUDA
    CUDA.device!(MPI.Comm_rank(MPI.COMM_WORLD) % length(CUDA.devices()))  # one GPU per rank
end

ctx = UCC.Context(UCC.MPIOOB(MPI.COMM_WORLD))
team = UCC.Team(ctx)
@test UCC.rank(team) == MPI.Comm_rank(MPI.COMM_WORLD)
@test UCC.size(team) == MPI.Comm_size(MPI.COMM_WORLD)

test_collectives(team, Array)
test_collectives(team, Array; T=Int64)

if TEST_CUDA
    test_collectives(team, CuArray)
    test_collectives(team, CuArray; T=Float16)
end

MPI.Comm_rank(MPI.COMM_WORLD) == 0 && println("MPI collectives OK on ", UCC.size(team), " ranks")
UCC.destroy(team)
UCC.destroy(ctx)
MPI.Finalize()
