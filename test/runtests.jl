using Test
using UCC
using UCC.LibUCC

# UCX's TCP transport probes reachability over netlink, which hangs on some
# machines; the tests only need shared memory transports.
get!(ENV, "UCX_TLS", "^tcp")

include("collectives.jl")

# run `f(team)` on every rank of an in-process team, one task per rank
function spmd(f, teams)
    tasks = [Threads.@spawn f(team) for team in teams]
    return fetch.(tasks)
end

# in-process ranks may run on different threads, so UCC must be thread-safe
local_context(oob) = UCC.Context(oob; lib=UCC.Library(thread_mode=:multiple))

function local_teams(n)
    spmd(UCC.LocalOOB(n)) do oob
        UCC.Team(local_context(oob))
    end
end

function destroy_all(teams)
    spmd(teams) do team
        ctx = UCC.context(team)
        UCC.destroy(team)
        UCC.destroy(ctx)
        UCC.destroy(ctx.lib)
    end
end

@testset "UCC" begin

include("layout.jl")

@testset "library" begin
    @test UCC.is_available()
    @test UCC.version() >= v"1.8"
    @test UCC.version_string() isa String

    cfg = UCC.LibraryConfig()
    @test occursin("UCC_", sprint(show, MIME"text/plain"(), cfg))
    @test_throws UCCError ucc_lib_config_modify(cfg, "NO_SUCH_OPTION", "1")
    UCC.release(cfg)

    lib = UCC.Library(; thread_mode=:multiple)
    @test UCC.thread_mode(lib) == UCC_THREAD_MULTIPLE
    @test UCC.attributes(lib).coll_types & UInt64(UCC_COLL_TYPE_ALLREDUCE) != 0

    ccfg = UCC.ContextConfig(lib)
    @test occursin("UCC_", sprint(show, MIME"text/plain"(), ccfg))
    UCC.release(ccfg)

    UCC.destroy(lib)
end

@testset "single rank" begin
    oob = UCC.LocalOOB(1)[1]
    ctx = UCC.Context(oob)
    @test UCC.work_buffer_size(ctx) >= 0
    @test !isempty(UCC.address(ctx))
    team = UCC.Team(ctx)
    @test UCC.rank(team) == 0
    @test UCC.size(team) == 1
    test_collectives(team, Array)
    @test_throws ArgumentError UCC.Allreduce!([true], +, team)
    @test_throws ArgumentError UCC.Allreduce!(Int128[1], +, team)
    @test_throws ArgumentError UCC.Allreduce!(Float16[1], +, team)
    UCC.destroy(team)
    @test team.handle == C_NULL
    UCC.destroy(ctx)
    @test ctx.lib.handle == C_NULL  # default library is destroyed with its context
    @test_throws ArgumentError UCC.Barrier(team)
end

@testset "deferred destruction" begin
    # finalizers only enqueue; nothing is freed until `free_deferred`
    let ctx = UCC.Context()
        finalize(ctx)
        @test ctx.handle != C_NULL
        UCC.free_deferred()
        @test ctx.handle == C_NULL
        @test ctx.lib.handle == C_NULL
    end
    # a parent is only freed after its children
    lib = UCC.Library()
    ctx = UCC.Context(; lib)
    UCC.destroy(lib)
    @test lib.handle != C_NULL
    UCC.destroy(ctx)
    @test lib.handle == C_NULL
end

@testset "in-process ranks (n=$n)" for n in (2, 3)
    teams = local_teams(n)
    @test sort(UCC.rank.(teams)) == 0:n-1
    @test all(==(n), UCC.size.(teams))
    spmd(team -> test_collectives(team, Array), teams)
    spmd(team -> test_collectives(team, Array; T=Int32), teams)
    spmd(team -> test_collectives(team, Array; T=ComplexF32), teams)

    # sub-team over a separate OOB
    sub_oobs = UCC.LocalOOB(n - 1)
    subteams = spmd(zip(teams[1:n-1], sub_oobs)) do (team, oob)
        UCC.Team(UCC.context(team), oob)
    end
    @test sort(UCC.rank.(subteams)) == 0:n-2
    spmd(subteams) do team
        x = fill(1.0, 2)
        UCC.Allreduce!(x, +, team)
        @test x == fill(Float64(n - 1), 2)
    end
    spmd(UCC.destroy, subteams)

    destroy_all(teams)
end

@testset "dropped in-flight request" begin
    teams = local_teams(2)
    req = UCC.Iallreduce!(ones(4), zeros(4), +, teams[1])
    finalize(req)
    UCC.free_deferred()
    @test req.handle != C_NULL  # other rank hasn't joined: stays deferred
    other = UCC.Iallreduce!(ones(4), zeros(4), +, teams[2])
    while req.handle != C_NULL
        UCC.test(other)
        UCC.free_deferred()
    end
    @test req.result == fill(2.0, 4)
    destroy_all(teams)
end

@testset "request and engine lifetimes" begin
    ctx = UCC.Context(UCC.LocalOOB(1)[1])

    # a team outlives its requests
    team = UCC.Team(ctx)
    req = UCC.Iallreduce!(ones(4), zeros(4), +, team)
    UCC.destroy(team)
    @test team.handle != C_NULL
    wait(req)
    @test team.handle == C_NULL

    # an engine outlives its triggered requests, and a team its engines
    team = UCC.Team(ctx)
    ee = UCC.ExecutionEngine(team, UCC_EE_CPU_THREAD, C_NULL)
    req = UCC.Iallreduce!(ones(4), zeros(4), +, team; post=false)
    UCC.triggered_post!(ee, req)
    UCC.destroy(ee)
    UCC.destroy(team)
    @test ee.handle != C_NULL && team.handle != C_NULL
    wait(req)
    @test ee.handle == C_NULL && team.handle == C_NULL

    UCC.destroy(ctx)
end

struct Point
    x::Int32
    y::Int32
end

@testset "generic datatype" begin
    dt = UCC.GenericDatatype(Point; op=(a, b) -> Point(a.x + b.x, max(a.y, b.y)))
    @test_throws ArgumentError UCC.GenericDatatype(Point)
    n = 3
    teams = local_teams(n)
    spmd(teams) do team
        r = UCC.rank(team)
        x = [Point(r, r), Point(1, -r)]
        y = similar(x)
        UCC.Allreduce!(x, y, +, team)
        @test y == [Point(n * (n - 1) ÷ 2, n - 1), Point(n, 0)]
        z = Vector{Point}(undef, 2n)
        UCC.Allgather!(x, z, team)
        @test z[1:2:end] == [Point(i, i) for i in 0:n-1]
    end
    destroy_all(teams)
    UCC.destroy(dt)
    @test_throws ArgumentError UCC._builtin_datatype(Point)
    dt = UCC.GenericDatatype(Point)
    @test_throws ArgumentError UCC._op(Point[], +)
    UCC.destroy(dt)
end

@testset "MPI" begin
    using MPI, CUDA
    nprocs = CUDA.functional() ? 2 : 3
    script = joinpath(@__DIR__, "mpi_collectives.jl")
    project = Base.active_project()
    cmd = `$(MPI.mpiexec()) -n $nprocs $(Base.julia_cmd()) --startup-file=no --project=$project $script`
    withenv("UCC_TEST_CUDA" => string(CUDA.functional())) do
        @test success(pipeline(cmd; stdout, stderr))
    end
end

end
