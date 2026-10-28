module LibUCC

# Low-level bindings to the UCC C API (ucc/api/ucc.h, ucc_def.h, ucc_status.h)
# as shipped by UCC_jll. Struct layouts mirror the C headers exactly; see
# test/layout.jl for the offset/size checks against the C compiler.

using UCC_jll
export UCC_jll

using CEnum: CEnum, @cenum

const NULL = C_NULL

ucc_bit(i) = UInt64(1) << i
ucc_mask(i) = ucc_bit(i) - UInt64(1)

#
# ucc_status.h
#

@cenum ucc_status_t::Int32 begin
    UCC_OK = 0
    UCC_INPROGRESS = 1
    UCC_OPERATION_INITIALIZED = 2
    UCC_ERR_NOT_SUPPORTED = -1
    UCC_ERR_NOT_IMPLEMENTED = -2
    UCC_ERR_INVALID_PARAM = -3
    UCC_ERR_NO_MEMORY = -4
    UCC_ERR_NO_RESOURCE = -5
    UCC_ERR_NO_MESSAGE = -6
    UCC_ERR_NOT_FOUND = -7
    UCC_ERR_TIMED_OUT = -8
    UCC_ERR_IO_ERROR = -9
    UCC_ERR_LAST = -100
end

function ucc_status_string(status)
    @ccall libucc.ucc_status_string(status::ucc_status_t)::Cstring
end

struct UCCError <: Exception
    code::ucc_status_t
end

function Base.showerror(io::IO, err::UCCError)
    return print(io, "UCCError: ", unsafe_string(ucc_status_string(err.code)),
        " (code ", Int(err.code), ", ", err.code, ")")
end

"""
    check(status)

Throw a [`UCCError`](@ref) if `status` is an error code (negative). Returns
`status` otherwise, so non-error codes such as `UCC_INPROGRESS` can be inspected
by the caller.
"""
function check(status::ucc_status_t)
    Int32(status) < 0 && throw(UCCError(status))
    return status
end

#
# ucc_def.h
#

mutable struct ucc_lib_info end
mutable struct ucc_context end
mutable struct ucc_team end
mutable struct ucc_mem_handle end
mutable struct ucc_lib_config end
mutable struct ucc_context_config end
mutable struct ucc_ee end
mutable struct ucc_dt_generic end

const ucc_lib_h = Ptr{ucc_lib_info}
const ucc_context_h = Ptr{ucc_context}
const ucc_team_h = Ptr{ucc_team}
const ucc_mem_h = Ptr{ucc_mem_handle}
const ucc_lib_config_h = Ptr{ucc_lib_config}
const ucc_context_config_h = Ptr{ucc_context_config}
const ucc_ee_h = Ptr{ucc_ee}
const ucc_dt_generic_t = ucc_dt_generic

struct ucc_coll_req
    status::ucc_status_t
end
const ucc_coll_req_t = ucc_coll_req
const ucc_coll_req_h = Ptr{ucc_coll_req}

struct ucc_coll_callback
    cb::Ptr{Cvoid}     # void (*)(void *data, ucc_status_t status)
    data::Ptr{Cvoid}
end
const ucc_coll_callback_t = ucc_coll_callback

const ucc_count_t = UInt64
const ucc_aint_t = UInt64
const ucc_coll_id_t = UInt16
const ucc_p2p_conn_t = Ptr{Cvoid}
const ucc_context_addr_h = Ptr{Cvoid}
const ucc_context_addr_len_t = Csize_t

@cenum ucc_config_print_flags_t::UInt32 begin
    UCC_CONFIG_PRINT_CONFIG = 1 << 0
    UCC_CONFIG_PRINT_HEADER = 1 << 1
    UCC_CONFIG_PRINT_DOC = 1 << 2
    UCC_CONFIG_PRINT_HIDDEN = 1 << 3
end

const UCC_DATATYPE_PREDEFINED = UInt64(0)
const UCC_DATATYPE_GENERIC = ucc_bit(0)
const UCC_DATATYPE_SHIFT = 3
const UCC_DATATYPE_CLASS_MASK = ucc_mask(UCC_DATATYPE_SHIFT)

ucc_predefined_dt(id) = (UInt64(id) << UCC_DATATYPE_SHIFT) | UCC_DATATYPE_PREDEFINED

#
# ucc_version.h
#

const UCC_VERSION_MAJOR_SHIFT = 24
const UCC_VERSION_MINOR_SHIFT = 16
UCC_VERSION(major, minor) = (major << UCC_VERSION_MAJOR_SHIFT) | (minor << UCC_VERSION_MINOR_SHIFT)
const UCC_API_MAJOR = 1
const UCC_API_MINOR = 8
const UCC_API_VERSION = UCC_VERSION(UCC_API_MAJOR, UCC_API_MINOR)

#
# ucc.h: datatypes and reductions
#

@cenum ucc_coll_type_t::UInt32 begin
    UCC_COLL_TYPE_ALLGATHER = 1 << 0
    UCC_COLL_TYPE_ALLGATHERV = 1 << 1
    UCC_COLL_TYPE_ALLREDUCE = 1 << 2
    UCC_COLL_TYPE_ALLTOALL = 1 << 3
    UCC_COLL_TYPE_ALLTOALLV = 1 << 4
    UCC_COLL_TYPE_BARRIER = 1 << 5
    UCC_COLL_TYPE_BCAST = 1 << 6
    UCC_COLL_TYPE_FANIN = 1 << 7
    UCC_COLL_TYPE_FANOUT = 1 << 8
    UCC_COLL_TYPE_GATHER = 1 << 9
    UCC_COLL_TYPE_GATHERV = 1 << 10
    UCC_COLL_TYPE_REDUCE = 1 << 11
    UCC_COLL_TYPE_REDUCE_SCATTER = 1 << 12
    UCC_COLL_TYPE_REDUCE_SCATTERV = 1 << 13
    UCC_COLL_TYPE_SCATTER = 1 << 14
    UCC_COLL_TYPE_SCATTERV = 1 << 15
    UCC_COLL_TYPE_LAST = (1 << 15) + 1
end

@cenum ucc_memory_type_t::UInt32 begin
    UCC_MEMORY_TYPE_HOST = 0
    UCC_MEMORY_TYPE_CUDA = 1
    UCC_MEMORY_TYPE_CUDA_MANAGED = 2
    UCC_MEMORY_TYPE_ROCM = 3
    UCC_MEMORY_TYPE_ROCM_MANAGED = 4
    UCC_MEMORY_TYPE_LAST = 5
end
const UCC_MEMORY_TYPE_UNKNOWN = UCC_MEMORY_TYPE_LAST

const ucc_datatype_t = UInt64

const UCC_DT_INT8 = ucc_predefined_dt(0)
const UCC_DT_INT16 = ucc_predefined_dt(1)
const UCC_DT_INT32 = ucc_predefined_dt(2)
const UCC_DT_INT64 = ucc_predefined_dt(3)
const UCC_DT_INT128 = ucc_predefined_dt(4)
const UCC_DT_UINT8 = ucc_predefined_dt(5)
const UCC_DT_UINT16 = ucc_predefined_dt(6)
const UCC_DT_UINT32 = ucc_predefined_dt(7)
const UCC_DT_UINT64 = ucc_predefined_dt(8)
const UCC_DT_UINT128 = ucc_predefined_dt(9)
const UCC_DT_FLOAT16 = ucc_predefined_dt(10)
const UCC_DT_FLOAT32 = ucc_predefined_dt(11)
const UCC_DT_FLOAT64 = ucc_predefined_dt(12)
const UCC_DT_BFLOAT16 = ucc_predefined_dt(13)
const UCC_DT_FLOAT128 = ucc_predefined_dt(14)
const UCC_DT_FLOAT32_COMPLEX = ucc_predefined_dt(15)
const UCC_DT_FLOAT64_COMPLEX = ucc_predefined_dt(16)
const UCC_DT_FLOAT128_COMPLEX = ucc_predefined_dt(17)
const UCC_DT_PREDEFINED_LAST = 18

# enum ucc_generic_dt_ops_field
const UCC_GENERIC_DT_OPS_FIELD_FLAGS = ucc_bit(0)

# ucc_generic_dt_ops_flags_t
const UCC_GENERIC_DT_OPS_FLAG_CONTIG = ucc_bit(0)
const UCC_GENERIC_DT_OPS_FLAG_REDUCE = ucc_bit(1)

struct ucc_reduce_cb_params
    mask::UInt64
    src1::Ptr{Cvoid}
    src2::Ptr{Cvoid}
    dst::Ptr{Cvoid}
    n_vectors::Csize_t
    count::Csize_t
    stride::Csize_t
    dt::Ptr{ucc_dt_generic_t}
    cb_ctx::Ptr{Cvoid}
end
const ucc_reduce_cb_params_t = ucc_reduce_cb_params

struct ucc_generic_dt_ops_reduce
    cb::Ptr{Cvoid}      # ucc_status_t (*)(const ucc_reduce_cb_params_t *)
    cb_ctx::Ptr{Cvoid}
end

struct ucc_generic_dt_ops
    mask::UInt64
    flags::UInt64
    contig_size::Csize_t
    start_pack::Ptr{Cvoid}    # void *(*)(void *context, const void *buffer, size_t count)
    start_unpack::Ptr{Cvoid}  # void *(*)(void *context, void *buffer, size_t count)
    packed_size::Ptr{Cvoid}   # size_t (*)(void *state)
    pack::Ptr{Cvoid}          # size_t (*)(void *state, size_t offset, void *dest, size_t max_length)
    unpack::Ptr{Cvoid}        # ucc_status_t (*)(void *state, size_t offset, const void *src, size_t length)
    finish::Ptr{Cvoid}        # void (*)(void *state)
    reduce::ucc_generic_dt_ops_reduce
end
const ucc_generic_dt_ops_t = ucc_generic_dt_ops

function ucc_dt_create_generic(ops, context, datatype_p)
    return check(
        @ccall libucc.ucc_dt_create_generic(ops::Ptr{ucc_generic_dt_ops_t},
            context::Ptr{Cvoid},
            datatype_p::Ptr{ucc_datatype_t})::ucc_status_t
    )
end

function ucc_dt_destroy(datatype)
    @ccall libucc.ucc_dt_destroy(datatype::ucc_datatype_t)::Cvoid
end

@cenum ucc_reduction_op_t::UInt32 begin
    UCC_OP_SUM = 0
    UCC_OP_PROD = 1
    UCC_OP_MAX = 2
    UCC_OP_MIN = 3
    UCC_OP_LAND = 4
    UCC_OP_LOR = 5
    UCC_OP_LXOR = 6
    UCC_OP_BAND = 7
    UCC_OP_BOR = 8
    UCC_OP_BXOR = 9
    UCC_OP_MAXLOC = 10
    UCC_OP_MINLOC = 11
    UCC_OP_AVG = 12
    UCC_OP_LAST = 13
end

#
# ucc.h: library
#

@cenum ucc_thread_mode_t::UInt32 begin
    UCC_THREAD_SINGLE = 0
    UCC_THREAD_FUNNELED = 1
    UCC_THREAD_MULTIPLE = 2
end

@cenum ucc_coll_sync_type_t::UInt32 begin
    UCC_NO_SYNC_COLLECTIVES = 0
    UCC_SYNC_COLLECTIVES = 1
end

# enum ucc_lib_params_field
const UCC_LIB_PARAM_FIELD_THREAD_MODE = ucc_bit(0)
const UCC_LIB_PARAM_FIELD_COLL_TYPES = ucc_bit(1)
const UCC_LIB_PARAM_FIELD_REDUCTION_TYPES = ucc_bit(2)
const UCC_LIB_PARAM_FIELD_SYNC_TYPE = ucc_bit(3)

# enum ucc_lib_attr_field
const UCC_LIB_ATTR_FIELD_THREAD_MODE = ucc_bit(0)
const UCC_LIB_ATTR_FIELD_COLL_TYPES = ucc_bit(1)
const UCC_LIB_ATTR_FIELD_REDUCTION_TYPES = ucc_bit(2)
const UCC_LIB_ATTR_FIELD_SYNC_TYPE = ucc_bit(3)

struct ucc_lib_params
    mask::UInt64
    thread_mode::ucc_thread_mode_t
    coll_types::UInt64
    reduction_types::UInt64
    sync_type::ucc_coll_sync_type_t
end
const ucc_lib_params_t = ucc_lib_params

struct ucc_lib_attr
    mask::UInt64
    thread_mode::ucc_thread_mode_t
    coll_types::UInt64
    reduction_types::UInt64
    sync_type::ucc_coll_sync_type_t
end
const ucc_lib_attr_t = ucc_lib_attr

function ucc_lib_config_read(env_prefix, filename, config)
    return check(
        @ccall libucc.ucc_lib_config_read(env_prefix::Cstring, filename::Cstring,
            config::Ptr{ucc_lib_config_h})::ucc_status_t
    )
end

function ucc_lib_config_release(config)
    @ccall libucc.ucc_lib_config_release(config::ucc_lib_config_h)::Cvoid
end

function ucc_lib_config_print(config, stream, title, print_flags)
    @ccall libucc.ucc_lib_config_print(config::ucc_lib_config_h, stream::Ptr{Cvoid},
        title::Cstring,
        print_flags::ucc_config_print_flags_t)::Cvoid
end

function ucc_lib_config_modify(config, name, value)
    return check(
        @ccall libucc.ucc_lib_config_modify(config::ucc_lib_config_h, name::Cstring,
            value::Cstring)::ucc_status_t
    )
end

function ucc_get_version(major_version, minor_version, release_number)
    @ccall libucc.ucc_get_version(major_version::Ptr{Cuint}, minor_version::Ptr{Cuint},
        release_number::Ptr{Cuint})::Cvoid
end

function ucc_get_version_string()
    @ccall libucc.ucc_get_version_string()::Cstring
end

function ucc_init_version(api_major_version, api_minor_version, params, config, lib_p)
    return check(
        @ccall libucc.ucc_init_version(api_major_version::Cuint,
            api_minor_version::Cuint,
            params::Ptr{ucc_lib_params_t},
            config::ucc_lib_config_h,
            lib_p::Ptr{ucc_lib_h})::ucc_status_t
    )
end

# `static inline` in ucc.h
function ucc_init(params, config, lib_p)
    return ucc_init_version(UCC_API_MAJOR, UCC_API_MINOR, params, config, lib_p)
end

function ucc_finalize(lib_p)
    return check(@ccall libucc.ucc_finalize(lib_p::ucc_lib_h)::ucc_status_t)
end

function ucc_lib_get_attr(lib_p, lib_attr)
    return check(
        @ccall libucc.ucc_lib_get_attr(lib_p::ucc_lib_h,
            lib_attr::Ptr{ucc_lib_attr_t})::ucc_status_t
    )
end

#
# ucc.h: context
#

@cenum ucc_context_type_t::UInt32 begin
    UCC_CONTEXT_EXCLUSIVE = 0
    UCC_CONTEXT_SHARED = 1
end

# enum ucc_context_params_field
const UCC_CONTEXT_PARAM_FIELD_TYPE = ucc_bit(0)
const UCC_CONTEXT_PARAM_FIELD_SYNC_TYPE = ucc_bit(1)
const UCC_CONTEXT_PARAM_FIELD_OOB = ucc_bit(2)
const UCC_CONTEXT_PARAM_FIELD_ID = ucc_bit(3)
const UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS = ucc_bit(4)

# enum ucc_context_attr_field
const UCC_CONTEXT_ATTR_FIELD_TYPE = ucc_bit(0)
const UCC_CONTEXT_ATTR_FIELD_SYNC_TYPE = ucc_bit(1)
const UCC_CONTEXT_ATTR_FIELD_CTX_ADDR = ucc_bit(2)
const UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN = ucc_bit(3)
const UCC_CONTEXT_ATTR_FIELD_WORK_BUFFER_SIZE = ucc_bit(4)

struct ucc_oob_coll
    # ucc_status_t (*)(void *src_buf, void *recv_buf, size_t size,
    #                  void *allgather_info, void **request)
    allgather::Ptr{Cvoid}
    req_test::Ptr{Cvoid}   # ucc_status_t (*)(void *request)
    req_free::Ptr{Cvoid}   # ucc_status_t (*)(void *request)
    coll_info::Ptr{Cvoid}
    n_oob_eps::UInt32
    oob_ep::UInt32
end
const ucc_oob_coll_t = ucc_oob_coll
const ucc_context_oob_coll_t = ucc_oob_coll_t
const ucc_team_oob_coll_t = ucc_oob_coll_t

struct ucc_mem_map
    address::Ptr{Cvoid}
    len::Csize_t
end
const ucc_mem_map_t = ucc_mem_map

struct ucc_mem_map_params
    segments::Ptr{ucc_mem_map_t}
    n_segments::UInt64
end
const ucc_mem_map_params_t = ucc_mem_map_params

struct ucc_context_params
    mask::UInt64
    type::ucc_context_type_t
    sync_type::ucc_coll_sync_type_t
    oob::ucc_context_oob_coll_t
    ctx_id::UInt64
    mem_params::ucc_mem_map_params_t
end
const ucc_context_params_t = ucc_context_params

struct ucc_context_attr
    mask::UInt64
    type::ucc_context_type_t
    sync_type::ucc_coll_sync_type_t
    ctx_addr::ucc_context_addr_h
    ctx_addr_len::ucc_context_addr_len_t
    global_work_buffer_size::UInt64
end
const ucc_context_attr_t = ucc_context_attr

function ucc_context_config_read(lib_handle, filename, config)
    return check(
        @ccall libucc.ucc_context_config_read(lib_handle::ucc_lib_h, filename::Cstring,
            config::Ptr{ucc_context_config_h})::ucc_status_t
    )
end

function ucc_context_config_release(config)
    @ccall libucc.ucc_context_config_release(config::ucc_context_config_h)::Cvoid
end

function ucc_context_config_print(config, stream, title, print_flags)
    @ccall libucc.ucc_context_config_print(config::ucc_context_config_h, stream::Ptr{Cvoid},
        title::Cstring,
        print_flags::ucc_config_print_flags_t)::Cvoid
end

function ucc_context_config_modify(config, component, name, value)
    return check(
        @ccall libucc.ucc_context_config_modify(config::ucc_context_config_h,
            component::Cstring, name::Cstring,
            value::Cstring)::ucc_status_t
    )
end

function ucc_context_create(lib_handle, params, config, context)
    return check(
        @ccall libucc.ucc_context_create(lib_handle::ucc_lib_h,
            params::Ptr{ucc_context_params_t},
            config::ucc_context_config_h,
            context::Ptr{ucc_context_h})::ucc_status_t
    )
end

function ucc_context_progress(context)
    return check(@ccall libucc.ucc_context_progress(context::ucc_context_h)::ucc_status_t)
end

function ucc_context_destroy(context)
    return check(@ccall libucc.ucc_context_destroy(context::ucc_context_h)::ucc_status_t)
end

function ucc_context_get_attr(context, context_attr)
    return check(
        @ccall libucc.ucc_context_get_attr(context::ucc_context_h,
            context_attr::Ptr{ucc_context_attr_t})::ucc_status_t
    )
end

#
# ucc.h: teams
#

# enum ucc_team_params_field
const UCC_TEAM_PARAM_FIELD_ORDERING = ucc_bit(0)
const UCC_TEAM_PARAM_FIELD_OUTSTANDING_COLLS = ucc_bit(1)
const UCC_TEAM_PARAM_FIELD_EP = ucc_bit(2)
const UCC_TEAM_PARAM_FIELD_EP_LIST = ucc_bit(3)
const UCC_TEAM_PARAM_FIELD_EP_RANGE = ucc_bit(4)
const UCC_TEAM_PARAM_FIELD_TEAM_SIZE = ucc_bit(5)
const UCC_TEAM_PARAM_FIELD_SYNC_TYPE = ucc_bit(6)
const UCC_TEAM_PARAM_FIELD_OOB = ucc_bit(7)
const UCC_TEAM_PARAM_FIELD_P2P_CONN = ucc_bit(8)
const UCC_TEAM_PARAM_FIELD_MEM_PARAMS = ucc_bit(9)
const UCC_TEAM_PARAM_FIELD_EP_MAP = ucc_bit(10)
const UCC_TEAM_PARAM_FIELD_ID = ucc_bit(11)
const UCC_TEAM_PARAM_FIELD_FLAGS = ucc_bit(12)

# enum ucc_team_attr_field
const UCC_TEAM_ATTR_FIELD_POST_ORDERING = ucc_bit(0)
const UCC_TEAM_ATTR_FIELD_OUTSTANDING_CALLS = ucc_bit(1)
const UCC_TEAM_ATTR_FIELD_EP = ucc_bit(2)
const UCC_TEAM_ATTR_FIELD_EP_RANGE = ucc_bit(3)
const UCC_TEAM_ATTR_FIELD_SYNC_TYPE = ucc_bit(4)
const UCC_TEAM_ATTR_FIELD_MEM_PARAMS = ucc_bit(5)
const UCC_TEAM_ATTR_FIELD_SIZE = ucc_bit(6)
const UCC_TEAM_ATTR_FIELD_EPS = ucc_bit(7)

# enum ucc_team_flags
const UCC_TEAM_FLAG_COLL_WORK_BUFFER = ucc_bit(0)

struct ucc_team_p2p_conn
    # int (*)(void *conn_ctx, uint64_t ep, ucc_p2p_conn_t **conn_info, void *request)
    conn_info_lookup::Ptr{Cvoid}
    conn_info_release::Ptr{Cvoid}  # int (*)(ucc_p2p_conn_t *conn_info)
    conn_ctx::Ptr{Cvoid}
    req_test::Ptr{Cvoid}           # ucc_status_t (*)(void *request)
    req_free::Ptr{Cvoid}           # ucc_status_t (*)(void *request)
end
const ucc_team_p2p_conn_t = ucc_team_p2p_conn

@cenum ucc_post_ordering_t::UInt32 begin
    UCC_COLLECTIVE_POST_ORDERED = 0
    UCC_COLLECTIVE_POST_UNORDERED = 1
    UCC_COLLECTIVE_INIT_ORDERED = 2
    UCC_COLLECTIVE_INIT_UNORDERED = 3
    UCC_COLLECTIVE_INIT_AND_POST_ORDERED = 4
    UCC_COLLECTIVE_INIT_AND_POST_UNORDERED = 5
end

@cenum ucc_ep_range_type_t::UInt32 begin
    UCC_COLLECTIVE_EP_RANGE_CONTIG = 0
    UCC_COLLECTIVE_EP_RANGE_NONCONTIG = 1
end

struct ucc_ep_map_strided
    start::UInt64
    stride::Int64
end

struct ucc_ep_map_array
    map::Ptr{Cvoid}
    elem_size::Csize_t
end

struct ucc_ep_map_cb
    cb::Ptr{Cvoid}   # uint64_t (*)(uint64_t ep, void *cb_ctx)
    cb_ctx::Ptr{Cvoid}
end

@cenum ucc_ep_map_type_t::UInt32 begin
    UCC_EP_MAP_FULL = 1
    UCC_EP_MAP_STRIDED = 2
    UCC_EP_MAP_ARRAY = 3
    UCC_EP_MAP_CB = 4
end

# The anonymous union {strided, array, cb} is stored as raw words; all
# members are two 8-byte fields.
struct ucc_ep_map_t
    type::ucc_ep_map_type_t
    ep_num::UInt64
    u::NTuple{2,UInt64}
end

function ucc_ep_map_t(type, ep_num, x::Union{ucc_ep_map_strided,ucc_ep_map_array,ucc_ep_map_cb})
    return ucc_ep_map_t(type, ep_num, reinterpret(NTuple{2,UInt64}, x))
end
ucc_ep_map_t(type, ep_num) = ucc_ep_map_t(type, ep_num, (UInt64(0), UInt64(0)))

struct ucc_team_params
    mask::UInt64
    flags::UInt64
    ordering::ucc_post_ordering_t
    outstanding_colls::UInt64
    ep::UInt64
    ep_list::Ptr{UInt64}
    ep_range::ucc_ep_range_type_t
    team_size::UInt64
    sync_type::ucc_coll_sync_type_t
    oob::ucc_team_oob_coll_t
    p2p_conn::ucc_team_p2p_conn_t
    mem_params::ucc_mem_map_params_t
    ep_map::ucc_ep_map_t
    id::UInt64
end
const ucc_team_params_t = ucc_team_params

struct ucc_team_attr
    mask::UInt64
    ordering::ucc_post_ordering_t
    outstanding_colls::UInt64
    ep::UInt64
    ep_range::ucc_ep_range_type_t
    sync_type::ucc_coll_sync_type_t
    mem_params::ucc_mem_map_params_t
    size::UInt32
    eps::Ptr{UInt64}
end
const ucc_team_attr_t = ucc_team_attr

function ucc_team_create_post(contexts, num_contexts, team_params, new_team)
    return check(
        @ccall libucc.ucc_team_create_post(contexts::Ptr{ucc_context_h},
            num_contexts::UInt32,
            team_params::Ptr{ucc_team_params_t},
            new_team::Ptr{ucc_team_h})::ucc_status_t
    )
end

function ucc_team_create_test(team)
    return check(@ccall libucc.ucc_team_create_test(team::ucc_team_h)::ucc_status_t)
end

function ucc_team_destroy(team)
    return check(@ccall libucc.ucc_team_destroy(team::ucc_team_h)::ucc_status_t)
end

function ucc_team_get_attr(team, team_attr)
    return check(
        @ccall libucc.ucc_team_get_attr(team::ucc_team_h,
            team_attr::Ptr{ucc_team_attr_t})::ucc_status_t
    )
end

function ucc_team_create_from_parent(my_ep, included, parent_team, new_team)
    return check(
        @ccall libucc.ucc_team_create_from_parent(my_ep::UInt64, included::UInt32,
            parent_team::ucc_team_h,
            new_team::Ptr{ucc_team_h})::ucc_status_t
    )
end

#
# ucc.h: collectives
#

# ucc_coll_args_flags_t
const UCC_COLL_ARGS_FLAG_IN_PLACE = ucc_bit(0)
const UCC_COLL_ARGS_FLAG_PERSISTENT = ucc_bit(1)
const UCC_COLL_ARGS_FLAG_COUNT_64BIT = ucc_bit(2)
const UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT = ucc_bit(3)
const UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER = ucc_bit(4)
const UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER = ucc_bit(5)
const UCC_COLL_ARGS_FLAG_TIMEOUT = ucc_bit(6)
const UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS = ucc_bit(7)
const UCC_COLL_ARGS_FLAG_SRC_MEMH_GLOBAL = ucc_bit(8)
const UCC_COLL_ARGS_FLAG_DST_MEMH_GLOBAL = ucc_bit(9)

# ucc_coll_args_hints_t
const UCC_COLL_ARGS_HINT_OPTIMIZE_OVERLAP_CPU = ucc_bit(24)
const UCC_COLL_ARGS_HINT_OPTIMIZE_OVERLAP_GPU = ucc_bit(25)
const UCC_COLL_ARGS_HINT_OPTIMIZE_LATENCY = ucc_bit(26)
const UCC_COLL_ARGS_HINT_CONTIG_SRC_BUFFER = UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER
const UCC_COLL_ARGS_HINT_CONTIG_DST_BUFFER = UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER

struct ucc_coll_buffer_info_v
    buffer::Ptr{Cvoid}
    counts::Ptr{ucc_count_t}
    displacements::Ptr{ucc_aint_t}
    datatype::ucc_datatype_t
    mem_type::ucc_memory_type_t
end
const ucc_coll_buffer_info_v_t = ucc_coll_buffer_info_v

struct ucc_coll_buffer_info
    buffer::Ptr{Cvoid}
    count::ucc_count_t
    datatype::ucc_datatype_t
    mem_type::ucc_memory_type_t
end
const ucc_coll_buffer_info_t = ucc_coll_buffer_info

# union { ucc_coll_buffer_info_t info; ucc_coll_buffer_info_v_t info_v; }
struct ucc_coll_buffer_info_u
    data::NTuple{5,UInt64}
end

ucc_coll_buffer_info_u() = ucc_coll_buffer_info_u(ntuple(_ -> UInt64(0), 5))
function ucc_coll_buffer_info_u(x::Union{ucc_coll_buffer_info_t,ucc_coll_buffer_info_v_t})
    r = Ref(ucc_coll_buffer_info_u())
    GC.@preserve r begin
        unsafe_store!(Ptr{typeof(x)}(Base.unsafe_convert(Ptr{ucc_coll_buffer_info_u}, r)), x)
    end
    return r[]
end
function Base.getproperty(u::ucc_coll_buffer_info_u, f::Symbol)
    f === :data && return getfield(u, :data)
    T = if f === :info
        ucc_coll_buffer_info_t
    elseif f === :info_v
        ucc_coll_buffer_info_v_t
    else
        error("type ucc_coll_buffer_info_u has no field $f")
    end
    r = Ref(u)
    GC.@preserve r unsafe_load(Ptr{T}(Base.unsafe_convert(Ptr{ucc_coll_buffer_info_u}, r)))
end

@cenum ucc_error_type_t::UInt32 begin
    UCC_ERR_TYPE_LOCAL = 0
    UCC_ERR_TYPE_GLOBAL = 1
end

# enum ucc_coll_args_field
const UCC_COLL_ARGS_FIELD_FLAGS = ucc_bit(0)
const UCC_COLL_ARGS_FIELD_TAG = ucc_bit(1)
const UCC_COLL_ARGS_FIELD_CB = ucc_bit(2)
const UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER = ucc_bit(3)
const UCC_COLL_ARGS_FIELD_ACTIVE_SET = ucc_bit(4)
const UCC_COLL_ARGS_FIELD_MEM_MAP_SRC_MEMH = ucc_bit(5)
const UCC_COLL_ARGS_FIELD_MEM_MAP_DST_MEMH = ucc_bit(6)

const ucc_mem_map_mem_h = Ptr{Cvoid}

struct ucc_coll_args_active_set
    start::UInt64
    stride::Int64
    size::UInt64
end

struct ucc_coll_args
    mask::UInt64
    coll_type::ucc_coll_type_t
    src::ucc_coll_buffer_info_u
    dst::ucc_coll_buffer_info_u
    op::ucc_reduction_op_t
    flags::UInt64
    root::UInt64
    error_type::ucc_error_type_t
    tag::ucc_coll_id_t
    global_work_buffer::Ptr{Cvoid}
    cb::ucc_coll_callback_t
    timeout::Cdouble
    active_set::ucc_coll_args_active_set
    # unions of {ucc_mem_map_mem_h local_memh; ucc_mem_map_mem_h *global_memh;}
    src_memh::Ptr{Cvoid}
    dst_memh::Ptr{Cvoid}
end
const ucc_coll_args_t = ucc_coll_args

function ucc_collective_init(coll_args, request, team)
    return check(
        @ccall libucc.ucc_collective_init(coll_args::Ptr{ucc_coll_args_t},
            request::Ptr{ucc_coll_req_h},
            team::ucc_team_h)::ucc_status_t
    )
end

function ucc_collective_post(request)
    return check(@ccall libucc.ucc_collective_post(request::ucc_coll_req_h)::ucc_status_t)
end

function ucc_collective_init_and_post(coll_args, request, team)
    return check(
        @ccall libucc.ucc_collective_init_and_post(coll_args::Ptr{ucc_coll_args_t},
            request::Ptr{ucc_coll_req_h},
            team::ucc_team_h)::ucc_status_t
    )
end

# `static inline` in ucc.h: returns request->status without checking it.
ucc_collective_test(request::ucc_coll_req_h) =
    unsafe_load(Ptr{ucc_status_t}(request))

function ucc_collective_finalize(request)
    return check(@ccall libucc.ucc_collective_finalize(request::ucc_coll_req_h)::ucc_status_t)
end

#
# ucc.h: execution engines and events
#

@cenum ucc_event_type_t::UInt32 begin
    UCC_EVENT_COLLECTIVE_POST = 1 << 0
    UCC_EVENT_COLLECTIVE_COMPLETE = 1 << 1
    UCC_EVENT_COMPUTE_COMPLETE = 1 << 2
    UCC_EVENT_OVERFLOW = 1 << 3
end

@cenum ucc_ee_type_t::UInt32 begin
    UCC_EE_CUDA_STREAM = 0
    UCC_EE_CPU_THREAD = 1
    UCC_EE_ROCM_STREAM = 2
    UCC_EE_LAST = 3
end
const UCC_EE_FIRST = UCC_EE_CUDA_STREAM
const UCC_EE_UNKNOWN = UCC_EE_LAST

struct ucc_event
    ev_type::ucc_event_type_t
    ev_context::Ptr{Cvoid}
    ev_context_size::Csize_t
    req::ucc_coll_req_h
end
const ucc_ev_t = ucc_event

struct ucc_ee_params
    ee_type::ucc_ee_type_t
    ee_context::Ptr{Cvoid}
    ee_context_size::Csize_t
end
const ucc_ee_params_t = ucc_ee_params

function ucc_ee_create(team, params, ee)
    return check(
        @ccall libucc.ucc_ee_create(team::ucc_team_h, params::Ptr{ucc_ee_params_t},
            ee::Ptr{ucc_ee_h})::ucc_status_t
    )
end

function ucc_ee_destroy(ee)
    return check(@ccall libucc.ucc_ee_destroy(ee::ucc_ee_h)::ucc_status_t)
end

# Returns UCC_OK if an event was retrieved, UCC_ERR_NOT_FOUND (an error code)
# if none is pending; callers distinguish the two, so this one is unchecked.
function ucc_ee_get_event(ee, ev)
    @ccall libucc.ucc_ee_get_event(ee::ucc_ee_h, ev::Ptr{Ptr{ucc_ev_t}})::ucc_status_t
end

function ucc_ee_ack_event(ee, ev)
    return check(@ccall libucc.ucc_ee_ack_event(ee::ucc_ee_h, ev::Ptr{ucc_ev_t})::ucc_status_t)
end

function ucc_ee_set_event(ee, ev)
    return check(@ccall libucc.ucc_ee_set_event(ee::ucc_ee_h, ev::Ptr{ucc_ev_t})::ucc_status_t)
end

function ucc_ee_wait(ee, ev)
    return check(@ccall libucc.ucc_ee_wait(ee::ucc_ee_h, ev::Ptr{ucc_ev_t})::ucc_status_t)
end

function ucc_collective_triggered_post(ee, ee_event)
    return check(
        @ccall libucc.ucc_collective_triggered_post(ee::ucc_ee_h,
            ee_event::Ptr{ucc_ev_t})::ucc_status_t
    )
end

#
# ucc.h: memory mapping
#

@cenum ucc_mem_map_mode_t::UInt32 begin
    UCC_MEM_MAP_MODE_EXPORT = 0
    UCC_MEM_MAP_MODE_IMPORT = 1
    UCC_MEM_MAP_MODE_EXPORT_OFFLOAD = 2
    UCC_MEM_MAP_MODE_IMPORT_OFFLOAD = 3
    UCC_MEM_MAP_MODE_LAST = 4
end

function ucc_mem_map(context, mode, params, memh_size, memh)
    return check(
        @ccall libucc.ucc_mem_map(context::ucc_context_h, mode::ucc_mem_map_mode_t,
            params::Ptr{ucc_mem_map_params_t},
            memh_size::Ptr{Csize_t},
            memh::Ptr{ucc_mem_map_mem_h})::ucc_status_t
    )
end

function ucc_mem_unmap(memh)
    return check(@ccall libucc.ucc_mem_unmap(memh::Ptr{ucc_mem_map_mem_h})::ucc_status_t)
end

# export all UCC_* / ucc_* symbols, like the Clang.jl-generated NCCL bindings
for name in names(@__MODULE__; all=true)
    s = string(name)
    if startswith(s, "ucc_") || startswith(s, "UCC_") || s == "UCCError"
        @eval export $name
    end
end

end # module LibUCC
