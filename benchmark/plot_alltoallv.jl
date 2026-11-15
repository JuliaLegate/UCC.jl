# Plot alltoallv.jl results into plots/alltoallv/ next to the CSV.
# Usage: julia --project=. plot_alltoallv.jl alltoallv.csv

ENV["GKSwstype"] = "100"
using CSV
using DataFrames
using Plots

csvfile = ARGS[1]
df = sort(CSV.read(csvfile, DataFrame), [:dist, :ngpus, :target_bytes])
outdir = mkpath(joinpath(dirname(abspath(csvfile)), "plots", "alltoallv"))
save(p, name) = savefig(p, joinpath(outdir, name))

labels = Dict("nccl" => "NCCL", "ucc_nccl" => "UCC+NCCL", "ucc_ucp" => "UCC+UCP")
backends = filter(in(df.backend), ["nccl", "ucc_nccl", "ucc_ucp"])
dists = unique(df.dist)
maxgpus = maximum(df.ngpus)
top = df[df.ngpus .== maxgpus, :]

# Per distribution: backends vs message size, and vs GPU count at the largest size
for g in groupby(top, :dist)
    dist = g.dist[1]
    t = plot(; xlabel="avg send per rank (bytes)", ylabel="time (us)", xscale=:log2,
        yscale=:log10, title="alltoallv $dist, $maxgpus GPUs", legend=:topleft)
    bw = plot(; xlabel="avg send per rank (bytes)", ylabel="bus bandwidth (GB/s)",
        xscale=:log2, title="alltoallv $dist, $maxgpus GPUs", legend=:topleft)
    for b in backends
        d = g[g.backend .== b, :]
        plot!(t, d.bytes, d.avg_us; label=labels[b], marker=:circle,
            ribbon=(d.avg_us .- d.min_us, d.time_us .- d.avg_us), fillalpha=0.2)
        plot!(bw, d.bytes, d.busbw_GBps; label=labels[b], marker=:circle)
    end
    save(t, "$(dist)_time_vs_size.png")
    save(bw, "$(dist)_busbw_vs_size.png")

    s = df[df.dist .== dist, :]
    big = maximum(s[s.ngpus .== maxgpus, :target_bytes])
    s = s[s.target_bytes .== big, :]
    tg = plot(; xlabel="GPUs", ylabel="time (ms)",
        title="alltoallv $dist, $(big ÷ 2^20) MiB/rank")
    for b in backends
        d = s[s.backend .== b, :]
        plot!(tg, d.ngpus, d.time_us ./ 1e3; label=labels[b], marker=:circle)
    end
    save(tg, "$(dist)_time_vs_gpus.png")
end

# Per backend: slowdown of each distribution vs uniform at the same average size
"uniform" in dists && for b in backends
    d = top[top.backend .== b, :]
    u = d[d.dist .== "uniform", [:target_bytes, :time_us]]
    p = plot(; xlabel="avg send per rank (bytes)", ylabel="time / uniform time",
        xscale=:log2, title="$(labels[b]) alltoallv, $maxgpus GPUs", legend=:topleft)
    for dist in filter(!=("uniform"), dists)
        j = innerjoin(d[d.dist .== dist, [:target_bytes, :time_us]], u;
            on=:target_bytes, makeunique=true)
        plot!(p, j.target_bytes, j.time_us ./ j.time_us_1; label=dist, marker=:circle)
    end
    hline!(p, [1]; color=:gray, linestyle=:dash, label="")
    save(p, "$(b)_vs_uniform.png")
end

# UCC+NCCL vs hand-written NCCL Send/Recv, per distribution
if all(in(backends), ("nccl", "ucc_nccl"))
    p = plot(; xlabel="avg send per rank (bytes)", ylabel="extra time vs NCCL (%)",
        xscale=:log2, title="alltoallv UCC+NCCL overhead, $maxgpus GPUs")
    for dist in dists
        d = top[top.dist .== dist, :]
        j = innerjoin(d[d.backend .== "nccl", [:target_bytes, :time_us]],
            d[d.backend .== "ucc_nccl", [:target_bytes, :time_us]];
            on=:target_bytes, renamecols="_nccl" => "_ucc")
        plot!(p, j.target_bytes, 100 .* (j.time_us_ucc ./ j.time_us_nccl .- 1);
            label=dist, marker=:circle)
    end
    save(p, "ucc_overhead.png")
end
