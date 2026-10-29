# Verify that the Julia struct layouts in LibUCC match the C headers shipped in
# UCC_jll, by compiling a small C program that prints sizeof/offsetof.
# Skipped when no C compiler is available.

using Test
using UCC.LibUCC
using UCC_jll: UCC_jll

# C struct name => (Julia type, C field names in Julia field order)
const LAYOUTS = [
    "ucc_coll_req_t" => (ucc_coll_req_t, ["status"]),
    "ucc_coll_callback_t" => (ucc_coll_callback_t, ["cb", "data"]),
    "ucc_reduce_cb_params_t" => (ucc_reduce_cb_params_t,
        ["mask", "src1", "src2", "dst", "n_vectors", "count", "stride", "dt", "cb_ctx"]),
    "ucc_generic_dt_ops_t" => (ucc_generic_dt_ops_t,
        ["mask", "flags", "contig_size", "start_pack", "start_unpack", "packed_size",
            "pack", "unpack", "finish", "reduce"]),
    "ucc_lib_params_t" => (ucc_lib_params_t,
        ["mask", "thread_mode", "coll_types", "reduction_types", "sync_type"]),
    "ucc_lib_attr_t" => (ucc_lib_attr_t,
        ["mask", "thread_mode", "coll_types", "reduction_types", "sync_type"]),
    "ucc_oob_coll_t" => (ucc_oob_coll_t,
        ["allgather", "req_test", "req_free", "coll_info", "n_oob_eps", "oob_ep"]),
    "ucc_mem_map_t" => (ucc_mem_map_t, ["address", "len"]),
    "ucc_mem_map_params_t" => (ucc_mem_map_params_t, ["segments", "n_segments"]),
    "ucc_context_params_t" => (ucc_context_params_t,
        ["mask", "type", "sync_type", "oob", "ctx_id", "mem_params"]),
    "ucc_context_attr_t" => (ucc_context_attr_t,
        ["mask", "type", "sync_type", "ctx_addr", "ctx_addr_len", "global_work_buffer_size"]),
    "ucc_team_p2p_conn_t" => (ucc_team_p2p_conn_t,
        ["conn_info_lookup", "conn_info_release", "conn_ctx", "req_test", "req_free"]),
    "ucc_ep_map_t" => (ucc_ep_map_t, ["type", "ep_num", "strided"]),
    "ucc_team_params_t" => (ucc_team_params_t,
        ["mask", "flags", "ordering", "outstanding_colls", "ep", "ep_list", "ep_range",
            "team_size", "sync_type", "oob", "p2p_conn", "mem_params", "ep_map", "id"]),
    "ucc_team_attr_t" => (ucc_team_attr_t,
        ["mask", "ordering", "outstanding_colls", "ep", "ep_range", "sync_type",
            "mem_params", "size", "eps"]),
    "ucc_coll_buffer_info_v_t" => (ucc_coll_buffer_info_v_t,
        ["buffer", "counts", "displacements", "datatype", "mem_type"]),
    "ucc_coll_buffer_info_t" => (ucc_coll_buffer_info_t,
        ["buffer", "count", "datatype", "mem_type"]),
    "ucc_coll_args_t" => (ucc_coll_args_t,
        ["mask", "coll_type", "src", "dst", "op", "flags", "root", "error_type", "tag",
            "global_work_buffer", "cb", "timeout", "active_set", "src_memh", "dst_memh"]),
    "ucc_ev_t" => (ucc_ev_t, ["ev_type", "ev_context", "ev_context_size", "req"]),
    "ucc_ee_params_t" => (ucc_ee_params_t, ["ee_type", "ee_context", "ee_context_size"]),
]

const CC = something(Sys.which("cc"), Sys.which("gcc"), Sys.which("clang"), Some(nothing))

if CC === nothing
    @info "No C compiler found; skipping UCC struct layout checks"
else
    @testset "struct layout matches ucc.h" begin
        mktempdir() do dir
            src = joinpath(dir, "layout.c")
            exe = joinpath(dir, "layout")
            open(src, "w") do io
                println(io, "#include <stdio.h>\n#include <stddef.h>\n#include <ucc/api/ucc.h>")
                println(io, "int main(void) {")
                for (cname, (_, fields)) in LAYOUTS
                    println(io, "  printf(\"%zu\", sizeof($cname));")
                    for f in fields
                        println(io, "  printf(\" %zu\", offsetof($cname, $f));")
                    end
                    println(io, "  printf(\"\\n\");")
                end
                println(io, "  return 0;\n}")
            end
            inc = joinpath(UCC_jll.artifact_dir, "include")
            run(`$CC -I$inc -o $exe $src`)
            lines = readlines(`$exe`)
            for ((cname, (T, fields)), line) in zip(LAYOUTS, lines)
                c = parse.(Int, split(line))
                jl = [sizeof(T); [Int(fieldoffset(T, i)) for i in 1:fieldcount(T)]]
                @test fieldcount(T) == length(fields)
                @test (cname, jl) == (cname, c)
            end
        end
    end
end
