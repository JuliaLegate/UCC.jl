# Plot collectives.jl results into plots/<coll>/ next to the CSV.
# Usage: julia --project=. plot.jl results.csv

ENV["GKSwstype"] = "100"  # no display needed
using CSV
using DataFrames
using Plots

csvfile = ARGS[1]
results = sort(CSV.read(csvfile, DataFrame), [:coll, :ngpus, :bytes])

labels = Dict("nccl" => "NCCL", "ucc_nccl" => "UCC+NCCL", "ucc_ucp" => "UCC+UCP")

for df in groupby(results, :coll)
    coll = df.coll[1]
    outdir = mkpath(joinpath(dirname(abspath(csvfile)), "plots", coll))
    save(p, name) = savefig(p, joinpath(outdir, name))
    backends = filter(in(df.backend), ["nccl", "ucc_nccl", "ucc_ucp"])
    maxgpus = maximum(df.ngpus)
    maxbytes = maximum(df[df.ngpus .== maxgpus, :bytes])
    sizename = "$(maxbytes ÷ 2^20) MiB"

    # Scaling with GPU count, at each count's largest message
    time_gpus = plot(; xlabel="GPUs", ylabel="time (ms)", title="$coll, ~$sizename")
    bw_gpus = plot(; xlabel="GPUs", ylabel="bus bandwidth (GB/s)", title="$coll, ~$sizename")
    for b in backends
        d = combine(groupby(df[df.backend .== b, :], :ngpus), sdf -> last(sdf))
        plot!(time_gpus, d.ngpus, d.time_us ./ 1e3; label=labels[b], marker=:circle)
        d = d[d.ngpus .> 1, :]  # bus bandwidth is 0 on one GPU
        plot!(bw_gpus, d.ngpus, d.busbw_GBps; label=labels[b], marker=:circle)
    end
    save(time_gpus, "time_vs_gpus.png")
    save(bw_gpus, "busbw_vs_gpus.png")

    # Message size sweep on all GPUs; band spans fastest to slowest rank
    time_size = plot(; xlabel="message size (bytes)", ylabel="time (us)", xscale=:log2,
        yscale=:log10, title="$coll, $maxgpus GPUs", legend=:topleft)
    bw_size = plot(; xlabel="message size (bytes)", ylabel="bus bandwidth (GB/s)",
        xscale=:log2, title="$coll, $maxgpus GPUs", legend=:topleft)
    for b in backends
        d = df[(df.backend .== b) .& (df.ngpus .== maxgpus), :]
        plot!(time_size, d.bytes, d.avg_us; label=labels[b], marker=:circle,
            ribbon=(d.avg_us .- d.min_us, d.time_us .- d.avg_us), fillalpha=0.2)
        plot!(bw_size, d.bytes, d.busbw_GBps; label=labels[b], marker=:circle)
    end
    save(time_size, "time_vs_size.png")
    save(bw_size, "busbw_vs_size.png")

    # UCC overhead over calling NCCL directly
    nccl = df[df.backend .== "nccl", [:ngpus, :bytes, :time_us]]
    ucc = df[df.backend .== "ucc_nccl", [:ngpus, :bytes, :time_us]]
    both = innerjoin(nccl, ucc; on=[:ngpus, :bytes], renamecols="_nccl" => "_ucc")
    isempty(both) && continue
    overhead = plot(; xlabel="message size (bytes)", ylabel="extra time vs NCCL (%)",
        xscale=:log2, title="$coll: UCC+NCCL overhead")
    for d in groupby(both, :ngpus)
        pct = 100 .* (d.time_us_ucc ./ d.time_us_nccl .- 1)
        plot!(overhead, d.bytes, pct; label="$(d.ngpus[1]) GPUs", marker=:circle)
    end
    save(overhead, "ucc_overhead.png")
end
