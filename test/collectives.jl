# Checks for every collective, run by each rank of `team` with buffers of
# array type `AT` (e.g. `Array` or `CuArray`). Shared by the in-process and MPI
# tests.

using Test
using UCC

function test_collectives(team::UCC.Team, AT; T=Float64)
    r = UCC.rank(team)
    n = UCC.size(team)
    host(x) = Array(x)

    # Allreduce
    x = AT(fill(T(r + 1), 5))
    y = AT(zeros(T, 5))
    @test UCC.Allreduce!(x, y, +, team) === y
    @test host(y) == fill(T(n * (n + 1) ÷ 2), 5)
    if T <: Real
        UCC.Allreduce!(x, max, team)
        @test host(x) == fill(T(n), 5)
        x = AT(fill(T(r + 1), 5))
        UCC.Allreduce!(x, y, min, team)
        @test host(y) == fill(T(1), 5)
    else
        @test_throws ArgumentError UCC.Allreduce!(x, max, team)
    end
    x = AT(fill(T(2), 3))
    UCC.Allreduce!(x, y[1:3], *, team) |> host |> z -> @test z == fill(T(2^n), 3)
    xi = AT(Int32[1 << r, 1])
    yi = AT(zeros(Int32, 2))
    UCC.Allreduce!(xi, yi, |, team)
    @test host(yi) == Int32[(1 << n) - 1, 1]
    @test_throws ArgumentError UCC.Allreduce!(AT(zeros(Float64, 2)), |, team)

    # Broadcast
    x = AT(r == n - 1 ? T[1, 2, 3] : zeros(T, 3))
    UCC.Broadcast!(x, team; root=n - 1)
    @test host(x) == T[1, 2, 3]
    s = AT(T[4, 5, 6])
    y = AT(zeros(T, 3))
    UCC.Broadcast!(r == 0 ? s : nothing, y, team; root=0)
    @test host(y) == T[4, 5, 6]

    # Reduce
    x = AT(fill(T(r + 1), 4))
    y = r == n - 1 ? AT(zeros(T, 4)) : nothing
    UCC.Reduce!(x, y, +, team; root=n - 1)
    r == n - 1 && @test host(y) == fill(T(n * (n + 1) ÷ 2), 4)
    UCC.Reduce!(x, +, team; root=0)
    r == 0 && @test host(x) == fill(T(n * (n + 1) ÷ 2), 4)

    # Allgather
    x = AT(fill(T(r), 2))
    y = AT(zeros(T, 2n))
    UCC.Allgather!(x, y, team)
    @test host(y) == T.(repeat(0:n-1, inner=2))
    y = zeros(T, 2n)
    y[2r+1:2r+2] .= r
    y = AT(y)
    UCC.Allgather!(y, team)
    @test host(y) == T.(repeat(0:n-1, inner=2))

    # Allgatherv: rank i contributes i+1 elements
    counts = 1:n
    x = AT(fill(T(r), r + 1))
    y = AT(zeros(T, sum(counts)))
    UCC.Allgatherv!(x, y, counts, team)
    @test host(y) == T.(reduce(vcat, [fill(i, i + 1) for i in 0:n-1]))

    # Alltoall: element j goes to rank j
    x = AT(T[100r + j for j in 0:n-1])
    y = AT(zeros(T, n))
    UCC.Alltoall!(x, y, team)
    @test host(y) == T[100j + r for j in 0:n-1]

    # Alltoallv: rank r sends j+1 copies of 100r+j to rank j
    x = AT(T.(reduce(vcat, [fill(100r + j, j + 1) for j in 0:n-1])))
    y = AT(zeros(T, n * (r + 1)))
    UCC.Alltoallv!(x, 1:n, y, fill(r + 1, n), team)
    @test host(y) == T.(reduce(vcat, [fill(100j + r, r + 1) for j in 0:n-1]))

    # Barrier, fan-in, fan-out
    @test UCC.Barrier(team) === nothing
    @test UCC.Fanin(team; root=n - 1) === nothing
    @test UCC.Fanout(team; root=0) === nothing

    # Gather
    x = AT(fill(T(r), 2))
    y = r == 0 ? AT(zeros(T, 2n)) : nothing
    UCC.Gather!(x, y, team; root=0)
    r == 0 && @test host(y) == T.(repeat(0:n-1, inner=2))

    # Gatherv
    x = AT(fill(T(r), r + 1))
    y = r == n - 1 ? AT(zeros(T, sum(1:n))) : nothing
    UCC.Gatherv!(x, y, 1:n, team; root=n - 1)
    r == n - 1 && @test host(y) == T.(reduce(vcat, [fill(i, i + 1) for i in 0:n-1]))

    # ReduceScatter: block r of sum_k (k + i)
    x = AT(T[r + i for i in 0:2n-1])
    y = AT(zeros(T, 2))
    UCC.ReduceScatter!(x, y, +, team)
    @test host(y) == T[n * i + n * (n - 1) ÷ 2 for i in 2r:2r+1]

    # ReduceScatterv: rank j receives j+1 elements
    x = AT(fill(T(r + 1), sum(1:n)))
    y = AT(zeros(T, r + 1))
    UCC.ReduceScatterv!(x, y, 1:n, +, team)
    @test host(y) == fill(T(n * (n + 1) ÷ 2), r + 1)

    # UCC's single-rank transport (tl/self) does not implement scatter(v)
    n > 1 && test_scatter(team, AT, T)

    test_nonblocking(team, AT, T)
    return nothing
end

function test_scatter(team, AT, T)
    r = UCC.rank(team)
    n = UCC.size(team)
    host(x) = Array(x)

    # Scatter
    x = r == 0 ? AT(T.(repeat(0:n-1, inner=2))) : nothing
    y = AT(zeros(T, 2))
    UCC.Scatter!(x, y, team; root=0)
    @test host(y) == fill(T(r), 2)

    # Scatterv
    x = r == n - 1 ? AT(T.(reduce(vcat, [fill(i, i + 1) for i in 0:n-1]))) : nothing
    y = AT(zeros(T, r + 1))
    UCC.Scatterv!(x, 1:n, y, team; root=n - 1)
    @test host(y) == fill(T(r), r + 1)
end

function test_nonblocking(team, AT, T)
    r = UCC.rank(team)
    n = UCC.size(team)
    host(x) = Array(x)

    # Nonblocking
    x = AT(fill(T(r + 1), 3))
    y = AT(zeros(T, 3))
    req = UCC.Iallreduce!(x, y, +, team)
    @test req isa UCC.Request
    while !UCC.test(req)
        yield()
    end
    @test host(y) == fill(T(n * (n + 1) ÷ 2), 3)
    @test wait(UCC.Ibarrier(team)) === nothing

    # Persistent
    x = AT(fill(T(1), 3))
    y = AT(zeros(T, 3))
    req = UCC.Iallreduce!(x, y, +, team; persistent=true)
    @test wait(req) === y
    @test host(y) == fill(T(n), 3)
    x .= 2
    wait(UCC.post!(req))
    @test host(y) == fill(T(2n), 3)
    UCC.free(req)

    # Errors
    @test_throws DimensionMismatch UCC.Allreduce!(AT(zeros(T, 2)), AT(zeros(T, 3)), +, team)
    @test_throws ArgumentError UCC.Ibroadcast!(AT(zeros(T, 2)), team; root=n)

    return nothing
end
