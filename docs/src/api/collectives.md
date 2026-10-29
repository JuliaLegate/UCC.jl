# Collectives

Blocking forms return the receive buffer; nonblocking `I…` forms return a
[`Request`](@ref) and accept `post`, `persistent`, `timeout` and `tag`.

## Reductions

```@docs
UCC.Allreduce!
UCC.Iallreduce!
UCC.Reduce!
UCC.Ireduce!
UCC.ReduceScatter!
UCC.IreduceScatter!
UCC.ReduceScatterv!
UCC.IreduceScatterv!
```

## Data movement

```@docs
UCC.Broadcast!
UCC.Ibroadcast!
UCC.Allgather!
UCC.Iallgather!
UCC.Allgatherv!
UCC.Iallgatherv!
UCC.Alltoall!
UCC.Ialltoall!
UCC.Alltoallv!
UCC.Ialltoallv!
UCC.Gather!
UCC.Igather!
UCC.Gatherv!
UCC.Igatherv!
UCC.Scatter!
UCC.Iscatter!
UCC.Scatterv!
UCC.Iscatterv!
```

## Synchronization

```@docs
UCC.Barrier
UCC.Ibarrier
UCC.Fanin
UCC.Ifanin
UCC.Fanout
UCC.Ifanout
```

## Buffers

```@docs
UCC.memory_type
UCC.synchronize_buffer
```
