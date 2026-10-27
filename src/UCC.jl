module UCC

using CEnum

include("libucc.jl")
using .LibUCC

include("base.jl")
include("oob.jl")
include("library.jl")
include("context.jl")
include("team.jl")
include("request.jl")
include("datatype.jl")
include("collective.jl")
include("memmap.jl")
include("ee.jl")

"""
    UCC.MPIOOB(comm::MPI.Comm) :: AbstractOOB

OOB transport over an MPI communicator. Available after `using MPI`.
"""
function MPIOOB end

end # module UCC
