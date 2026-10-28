# Object Lifetimes

Objects form a hierarchy: `Library` ⊃ `Context` ⊃ `Team` ⊃ requests and
execution engines. A parent is only freed after all its children.

- Call [`UCC.destroy`](@ref) when done. If children are still alive, the parent
  is freed when the last one goes away.
- Teams are never freed by the GC: destroying a team is collective, so call
  `UCC.destroy(team)` on all ranks.
- Finalizers never call into UCC (they may run on any thread). Garbage-collected
  objects are queued until you call [`UCC.free_deferred`](@ref) at a point where
  no other thread is using UCC.
- A nonblocking request that you drop keeps its buffers alive until it
  completes and `free_deferred` releases it.
