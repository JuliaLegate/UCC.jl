# Plot allreduce.jl results. Writes PNGs next to the CSV.
# Usage: julia --project plot.jl results.csv

ENV["GKSwstype"] = "100"  # no display needed
using CSV
using DataFrames
using Plots

csvfile = ARGS[1]
outdir = dirname(abspath(csvfile))
df = sort(CSV.read(csvfile, DataFrame), [:ngpus, :bytes])

labels = Dict("nccl" => "NCCL", "ucc_nccl" => "UCC+NCCL", "ucc_ucp" => "UCC+UCP")
backends = filter(in(df.backend), ["nccl", "ucc_nccl", "ucc_ucp"])
maxgpus = maximum(df.ngpus)
maxbytes = maximum(df.bytes)
sizename = "$(maxbytes ÷ 2^20) MiB"

save(p, name) = savefig(p, joinpath(outdir, name))

# Scaling with GPU count, at the largest message
time_gpus = plot(; xlabel="GPUs", ylabel="time (ms)", title="Allreduce, $sizename")
bw_gpus = plot(; xlabel="GPUs", ylabel="bus bandwidth (GB/s)", title="Allreduce, $sizename")
for b in backends
    d = df[(df.backend .== b) .& (df.bytes .== maxbytes), :]
    plot!(time_gpus, d.ngpus, d.time_us ./ 1e3; label=labels[b], marker=:circle)
    d = d[d.ngpus .> 1, :]  # bus bandwidth is 0 on one GPU
    plot!(bw_gpus, d.ngpus, d.busbw_GBps; label=labels[b], marker=:circle)
end
save(time_gpus, "time_vs_gpus.png")
save(bw_gpus, "busbw_vs_gpus.png")

# Message size sweep, on all GPUs
time_size = plot(; xlabel="message size (bytes)", ylabel="time (us)", xscale=:log2,
    yscale=:log10, title="Allreduce, $maxgpus GPUs", legend=:topleft)
bw_size = plot(; xlabel="message size (bytes)", ylabel="bus bandwidth (GB/s)",
    xscale=:log2, title="Allreduce, $maxgpus GPUs", legend=:topleft)
for b in backends
    d = df[(df.backend .== b) .& (df.ngpus .== maxgpus), :]
    plot!(time_size, d.bytes, d.time_us; label=labels[b], marker=:circle)
    plot!(bw_size, d.bytes, d.busbw_GBps; label=labels[b], marker=:circle)
end
save(time_size, "time_vs_size.png")
save(bw_size, "busbw_vs_size.png")

# UCC overhead over calling NCCL directly
nccl = df[df.backend .== "nccl", [:ngpus, :bytes, :time_us]]
ucc = df[df.backend .== "ucc_nccl", [:ngpus, :bytes, :time_us]]
both = innerjoin(nccl, ucc; on=[:ngpus, :bytes], renamecols="_nccl" => "_ucc")
overhead = plot(; xlabel="message size (bytes)", ylabel="extra time vs NCCL (%)",
    xscale=:log2, title="UCC+NCCL overhead")
for d in groupby(both, :ngpus)
    pct = 100 .* (d.time_us_ucc ./ d.time_us_nccl .- 1)
    plot!(overhead, d.bytes, pct; label="$(d.ngpus[1]) GPUs", marker=:circle)
end
save(overhead, "ucc_overhead.png")
