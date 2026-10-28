using Documenter, DocumenterVitepress
using UCC

# DocumenterVitepress hardcodes nested sidebar groups as collapsed: false.
@eval DocumenterVitepress function pagelist2str(doc, name_contents::Pair{String,<:AbstractVector})
    name, contents = name_contents
    rendered_contents = pagelist2str.((doc,), contents)
    return "{ text: '$(replace(name, "'" => "\\'"))', collapsed: true, items: [\n$(join(rendered_contents, ",\n"))]\n }"
end

makedocs(;
    sitename="UCC.jl",
    authors="David Krasowska and Ethan Meitz",
    format=MarkdownVitepress(;
        repo="github.com/JuliaLegate/UCC.jl", devbranch="main", devurl="stable"
    ),
    pages=[
        "Home" => "index.md",
        "Guide" => [
            "Bootstrapping (OOB)" => "guide/oob.md",
            "Object Lifetimes" => "guide/lifetimes.md",
            "GPU Buffers" => "guide/gpu.md",
        ],
        "API" => [
            "Library and Context" => "api/library.md",
            "Teams" => "api/teams.md",
            "Collectives" => "api/collectives.md",
            "Requests" => "api/requests.md",
            "Datatypes and Operators" => "api/datatypes.md",
            "Out-of-Band" => "api/oob.md",
            "Memory Mapping" => "api/memmap.md",
            "Execution Engines" => "api/execution_engines.md",
        ],
    ],
)

if get(ENV, "CI", "") == "true"
    DocumenterVitepress.deploydocs(;
        repo="github.com/JuliaLegate/UCC.jl",
        target=joinpath(@__DIR__, "build"),
        branch="gh-pages",
        devbranch="main",
        push_preview=true,
    )
end
