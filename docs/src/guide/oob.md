# Bootstrapping (OOB)

UCC creates contexts and teams with an out-of-band (OOB) allgather that you
provide. UCC.jl ships two:

- [`UCC.MPIOOB`](@ref)`(comm)`: over an MPI communicator (after `using MPI`).
- [`UCC.LocalOOB`](@ref)`(n)`: `n` ranks as tasks in one process, for testing.

```julia
oobs  = UCC.LocalOOB(2)
teams = fetch.([Threads.@spawn UCC.Team(UCC.Context(o; lib=UCC.Library(thread_mode=:multiple)))
                for o in oobs])
```

In-process ranks may run on different threads, so *every* library in the
process must use `thread_mode=:multiple`. Start Julia with `-t` at least the
number of ranks.

For another transport, subtype [`UCC.AbstractOOB`](@ref) and implement
`oob_rank`, `oob_size` and `allgather`.

A team can use an OOB over a subset of the context's ranks:
`UCC.Team(ctx, sub_oob)`.
