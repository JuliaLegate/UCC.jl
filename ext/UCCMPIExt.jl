module UCCMPIExt

using UCC
using MPI

mutable struct MPIOOB <: UCC.AbstractOOB
    const comm::MPI.Comm
end

UCC.MPIOOB(comm::MPI.Comm) = MPIOOB(comm)

UCC.oob_rank(oob::MPIOOB) = MPI.Comm_rank(oob.comm)
UCC.oob_size(oob::MPIOOB) = MPI.Comm_size(oob.comm)
UCC.allgather(oob::MPIOOB, data::Vector{UInt8}) = MPI.Allgather(data, oob.comm)

end # module UCCMPIExt
