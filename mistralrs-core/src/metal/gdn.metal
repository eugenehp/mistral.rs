//! GatedDeltaNet (GDN) Metal kernels for Qwen3.5 / Qwen3Next hybrid models.
//!
//! Provides Metal equivalents of the CUDA kernels in src/cuda/gdn.cu:
//!   1. gdn_recurrence       – gated delta rule recurrence (f32)
//!   2. gdn_conv1d_update    – single-step causal conv1d (decode)
//!   3. gdn_conv1d_full      – full-sequence causal conv1d (prefill)
//!   4. gdn_save_conv_state  – save last kernel_size positions to conv state
//!   5. gdn_gating           – fused beta = sigmoid(b), g = -exp(a_log)*softplus(a+dt)

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Kernel 1: gated_delta_rule_recurrence (float32)
//
// V-tiled: each threadgroup processes one V-tile of the output for one (B,H).
// Grid:   (ceil(V/BV), B*H)   Threadgroup: (BV,)
//
// q, k  : [BH, S, K]
// v     : [BH, S, V]
// g, beta:[BH, S]
// state : [BH, K, V]  – modified in-place (initial → final state)
// output: [BH, S, V]
//
// Implementation note: state is read/written directly via device memory at
// each time step, avoiding any private-array aliasing or compiler reordering
// issues that arise from dynamically-indexed thread-local arrays.
// ============================================================================

#define GDN_BV    64u
// Maximum supported key-head dimension.  Raising this costs registers but
// never affects correctness; lower values keep occupancy higher.
// Qwen3.5-0.6B uses K=64, Qwen3.5-7B uses K=128.  256 covers all variants.
#define GDN_MAX_K 256u

kernel void gdn_recurrence(
    device const float* q         [[buffer(0)]],
    device const float* k         [[buffer(1)]],
    device const float* v         [[buffer(2)]],
    device const float* g         [[buffer(3)]],
    device const float* beta      [[buffer(4)]],
    device float*       state     [[buffer(5)]],
    device float*       output    [[buffer(6)]],
    constant uint&      seq_len   [[buffer(7)]],
    constant uint&      k_dim     [[buffer(8)]],
    constant uint&      v_dim     [[buffer(9)]],
    constant float&     q_scale   [[buffer(10)]],
    constant uint&      num_heads [[buffer(11)]],     // H; for [B,S,H,V] output layout
    threadgroup float*  shared    [[threadgroup(0)]], // 2 * k_dim floats
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]])
{
    const uint v_tile  = tgpig.x;
    const uint bh      = tgpig.y;
    const uint tid     = tpitg.x;
    const uint v_idx   = v_tile * GDN_BV + tid;
    // NOTE: do NOT early-return here – threads beyond v_dim must still
    // participate in all threadgroup_barrier() calls to avoid deadlock.
    const bool valid   = (v_idx < v_dim);

    const uint K = k_dim;
    const uint V = v_dim;
    const uint H = num_heads;

    const uint b_idx = bh / H;
    const uint h_idx = bh % H;

    device const float* q_bh    = q    + bh * seq_len * K;
    device const float* k_bh    = k    + bh * seq_len * K;
    device const float* v_bh    = v    + bh * seq_len * V;
    device const float* g_bh    = g    + bh * seq_len;
    device const float* beta_bh = beta + bh * seq_len;
    device float*       st_bh   = state  + bh * K * V;
    // Output in [B, S, H, V] layout: stride H*V per time step.
    device float* out_row = output + b_idx * seq_len * H * V + h_idx * V;

    // Shared memory layout: k_buf[K] | q_buf[K]
    threadgroup float* k_buf = shared;
    threadgroup float* q_buf = shared + K;

    // ── Register-resident state ──────────────────────────────────────────────
    // Each thread owns one column of the [K, V] state matrix (index v_idx).
    // Loading K floats into a thread-local array eliminates the O(T × K)
    // device-memory traffic of the original per-step read/write approach,
    // replacing it with a single load + single store of K floats.
    // On M3 Max, K ≤ 128 fits comfortably in the register file at BV=64.
    float s[GDN_MAX_K];
    if (valid) {
        for (uint j = 0; j < K; j++)
            s[j] = st_bh[j * V + v_idx];
    }

    for (uint t = 0; t < seq_len; t++) {
        // ---- Step A: cooperatively load k_t --------------------------------
        for (uint j = tid; j < K; j += GDN_BV) {
            k_buf[j] = k_bh[t * K + j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float decay  = valid ? exp(g_bh[t])         : 1.0f;
        float beta_t = valid ? beta_bh[t]            : 0.0f;
        float v_t    = valid ? v_bh[t * V + v_idx]  : 0.0f;

        // ---- Pass 1: decay register state, accumulate kv_mem ---------------
        float kv_mem = 0.0f;
        if (valid) {
            for (uint j = 0; j < K; j++) {
                s[j] *= decay;
                kv_mem = fma(s[j], k_buf[j], kv_mem);
            }
        }

        float delta = (v_t - kv_mem) * beta_t;

        // ---- Step B: cooperatively load q_t (scaled) -----------------------
        for (uint j = tid; j < K; j += GDN_BV) {
            q_buf[j] = q_bh[t * K + j] * q_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Pass 2: update register state, accumulate output --------------
        float y_t = 0.0f;
        if (valid) {
            for (uint j = 0; j < K; j++) {
                s[j] = fma(k_buf[j], delta, s[j]);
                y_t  = fma(s[j], q_buf[j], y_t);
            }
            out_row[t * H * V + v_idx] = y_t;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write register state back to device memory once (was: every step).
    if (valid) {
        for (uint j = 0; j < K; j++)
            st_bh[j * V + v_idx] = s[j];
    }
}

// ============================================================================
// Kernel 1b: gdn_recurrence_typed
//
// Same algorithm as gdn_recurrence but accepts native-dtype (half/bfloat)
// inputs for q, k, v, g, beta, converting to F32 at load time.
// Eliminates 5 separate to_dtype(F32) GPU kernels per decode step per layer.
//
// q, k  : [BH, S, K]  in InT
// v     : [BH, S, V]  in InT
// g, beta: [BH, S]    in InT
// state : [BH, K, V]  in float  — mutated in-place
// output: [BH, S, V]  in float
// ============================================================================
// ── gdn_recurrence_typed_impl ────────────────────────────────────────────────
//
// Changes vs. the naive typed kernel:
//   • output is InT (native dtype, not float) — eliminates the to_dtype(F32)
//     dispatch after the recurrence call.
//   • output layout is [B, S, H, V] — eliminates the transpose(1,2)+contiguous
//     dispatch in the Rust return path.
//   • num_heads is a new parameter (buffer 11) used to compute B and H from bh.
//
// The output global index for (bh, t, v_idx) is:
//   b  = bh / num_heads
//   h  = bh % num_heads
//   out[b * S*H*V + t * H*V + h*V + v_idx]
// ─────────────────────────────────────────────────────────────────────────────
template <typename InT>
kernel void gdn_recurrence_typed_impl(
    device const InT*   q         [[buffer(0)]],
    device const InT*   k         [[buffer(1)]],
    device const InT*   v         [[buffer(2)]],
    device const InT*   g         [[buffer(3)]],
    device const InT*   beta      [[buffer(4)]],
    device float*       state     [[buffer(5)]],
    device InT*         output    [[buffer(6)]],   // native dtype, [B,S,H,V] layout
    constant uint&      seq_len   [[buffer(7)]],
    constant uint&      k_dim     [[buffer(8)]],
    constant uint&      v_dim     [[buffer(9)]],
    constant float&     q_scale   [[buffer(10)]],
    constant uint&      num_heads [[buffer(11)]],  // H; used to compute B, h from bh
    threadgroup float*  shared    [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]])
{
    const uint v_tile = tgpig.x;
    const uint bh     = tgpig.y;
    const uint tid    = tpitg.x;
    const uint v_idx  = v_tile * GDN_BV + tid;
    const bool valid  = (v_idx < v_dim);

    const uint K = k_dim;
    const uint V = v_dim;
    const uint H = num_heads;

    // Decompose bh → batch b, head h for the [B, S, H, V] output layout.
    const uint b_idx = bh / H;
    const uint h_idx = bh % H;

    device const InT*  q_bh    = q    + bh * seq_len * K;
    device const InT*  k_bh    = k    + bh * seq_len * K;
    device const InT*  v_bh    = v    + bh * seq_len * V;
    device const InT*  g_bh    = g    + bh * seq_len;
    device const InT*  beta_bh = beta + bh * seq_len;
    device float*      st_bh   = state + bh * K * V;
    // Output row stride in the [B, S, H, V] layout: H*V elements per time step.
    device InT*  out_row = output + b_idx * seq_len * H * V + h_idx * V;

    threadgroup float* k_buf = shared;
    threadgroup float* q_buf = shared + K;

    // Register-resident state: one column of [K, V] per thread.
    float s[GDN_MAX_K];
    if (valid) {
        for (uint j = 0; j < K; j++)
            s[j] = st_bh[j * V + v_idx];
    }

    for (uint t = 0; t < seq_len; t++) {
        for (uint j = tid; j < K; j += GDN_BV) {
            k_buf[j] = float(k_bh[t * K + j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float decay  = valid ? exp(float(g_bh[t]))        : 1.0f;
        float beta_t = valid ? float(beta_bh[t])           : 0.0f;
        float v_t    = valid ? float(v_bh[t * V + v_idx]) : 0.0f;

        float kv_mem = 0.0f;
        if (valid) {
            for (uint j = 0; j < K; j++) {
                s[j] *= decay;
                kv_mem = fma(s[j], k_buf[j], kv_mem);
            }
        }

        float delta = (v_t - kv_mem) * beta_t;

        for (uint j = tid; j < K; j += GDN_BV) {
            q_buf[j] = float(q_bh[t * K + j]) * q_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float y_t = 0.0f;
        if (valid) {
            for (uint j = 0; j < K; j++) {
                s[j] = fma(k_buf[j], delta, s[j]);
                y_t  = fma(s[j], q_buf[j], y_t);
            }
            // Write directly into [B, S, H, V] position: stride H*V per time step.
            out_row[t * H * V + v_idx] = InT(y_t);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid) {
        for (uint j = 0; j < K; j++)
            st_bh[j * V + v_idx] = s[j];
    }
}

#define INST_RECURRENCE_TYPED(T, suffix)                                            \
    template [[host_name("gdn_recurrence_typed_" #suffix)]] [[kernel]] void         \
    gdn_recurrence_typed_impl<T>(                                                   \
        device const T*   q         [[buffer(0)]],                                  \
        device const T*   k         [[buffer(1)]],                                  \
        device const T*   v         [[buffer(2)]],                                  \
        device const T*   g         [[buffer(3)]],                                  \
        device const T*   beta      [[buffer(4)]],                                  \
        device float*     state     [[buffer(5)]],                                  \
        device T*         output    [[buffer(6)]],                                  \
        constant uint&    seq_len   [[buffer(7)]],                                  \
        constant uint&    k_dim     [[buffer(8)]],                                  \
        constant uint&    v_dim     [[buffer(9)]],                                  \
        constant float&   q_scale   [[buffer(10)]],                                 \
        constant uint&    num_heads [[buffer(11)]],                                 \
        threadgroup float* shared   [[threadgroup(0)]],                             \
        uint3 tgpig [[threadgroup_position_in_grid]],                               \
        uint3 tpitg [[thread_position_in_threadgroup]]);

INST_RECURRENCE_TYPED(half,   half)
#if __METAL_VERSION__ >= 310
INST_RECURRENCE_TYPED(bfloat, bfloat)
#endif

// ============================================================================
// Kernel 2: causal_conv1d_update  (decode path, one new token)
//
// Each thread processes one (batch, channel) pair.
// Grid: (ceil(conv_dim / BT), batch_size)   Threadgroup: (BT,)
//
// x          : [B, conv_dim]          (single token – squeezed)
// weight     : [conv_dim, kernel_size]
// conv_state : [B, conv_dim, kernel_size]  (in-place update)
// output     : [B, conv_dim]
// ============================================================================

// x_row_stride: the stride (in elements) between batches in the x buffer.
// For a standard contiguous [B, conv_dim] input this equals conv_dim.
// For a non-contiguous narrow view (e.g. from the FusedAll projection output
// [B, 1, total_out_dim]), this equals total_out_dim — so the kernel can read
// the correct channel for each batch without a prior transpose+contiguous copy.
template <typename T>
kernel void gdn_conv1d_update_impl(
    device const T* x             [[buffer(0)]],
    device const T* weight        [[buffer(1)]],
    device T*       conv_state    [[buffer(2)]],
    device T*       output        [[buffer(3)]],
    constant uint&  conv_dim      [[buffer(4)]],
    constant uint&  kernel_size   [[buffer(5)]],
    constant uint&  x_row_stride  [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint ch = gid.x;
    const uint b  = gid.y;
    if (ch >= conv_dim) return;

    device T*       cs = conv_state + (b * conv_dim + ch) * kernel_size;
    device const T* w  = weight + ch * kernel_size;

    // Shift ring buffer left by 1, insert new token value at the end.
    for (uint i = 0; i < kernel_size - 1; i++) {
        cs[i] = cs[i + 1];
    }
    // Read using the actual batch stride rather than assuming conv_dim.
    cs[kernel_size - 1] = x[b * x_row_stride + ch];

    // Dot product with weight, then SiLU.
    float acc = 0.0f;
    for (uint i = 0; i < kernel_size; i++) {
        acc = fma(float(cs[i]), float(w[i]), acc);
    }
    float sig = 1.0f / (1.0f + exp(-acc));
    output[b * conv_dim + ch] = T(acc * sig);
}

#define INST_CONV1D_UPDATE(T, suffix)                                               \
    template [[host_name("gdn_conv1d_update_" #suffix)]] [[kernel]] void            \
    gdn_conv1d_update_impl<T>(                                                      \
        device const T* x             [[buffer(0)]],                                \
        device const T* weight        [[buffer(1)]],                                \
        device T*       conv_state    [[buffer(2)]],                                \
        device T*       output        [[buffer(3)]],                                \
        constant uint&  conv_dim      [[buffer(4)]],                                \
        constant uint&  kernel_size   [[buffer(5)]],                                \
        constant uint&  x_row_stride  [[buffer(6)]],                                \
        uint2 gid [[thread_position_in_grid]]);

INST_CONV1D_UPDATE(float, float)
INST_CONV1D_UPDATE(half,  half)
#if __METAL_VERSION__ >= 310
INST_CONV1D_UPDATE(bfloat, bfloat)
#endif

// ============================================================================
// Kernel 3a: causal_conv1d_full  (prefill path, full sequence)
//
// Grid: (ceil(conv_dim/BT), seq_len, batch_size)   Threadgroup: (BT, 1, 1)
//
// x      : [B, conv_dim, S]
// weight : [conv_dim, kernel_size]
// output : [B, conv_dim, S]
// ============================================================================

template <typename T>
kernel void gdn_conv1d_full_impl(
    device const T* x           [[buffer(0)]],
    device const T* weight      [[buffer(1)]],
    device T*       output      [[buffer(2)]],
    constant uint&  conv_dim    [[buffer(3)]],
    constant uint&  seq_len     [[buffer(4)]],
    constant uint&  kernel_size [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint ch  = gid.x;
    const uint pos = gid.y;
    const uint b   = gid.z;

    if (ch >= conv_dim || pos >= seq_len) return;

    device const T* x_bch = x + (b * conv_dim + ch) * seq_len;
    device const T* w     = weight + ch * kernel_size;

    // Causal convolution: window ending at pos (zero-pad for positions < 0)
    float acc = 0.0f;
    for (uint i = 0; i < kernel_size; i++) {
        int src = (int)pos - (int)(kernel_size - 1) + (int)i;
        float xv = (src >= 0) ? float(x_bch[src]) : 0.0f;
        acc = fma(xv, float(w[i]), acc);
    }
    float sig = 1.0f / (1.0f + exp(-acc));
    output[(b * conv_dim + ch) * seq_len + pos] = T(acc * sig);
}

#define INST_CONV1D_FULL(T, suffix)                                                 \
    template [[host_name("gdn_conv1d_full_" #suffix)]] [[kernel]] void              \
    gdn_conv1d_full_impl<T>(                                                        \
        device const T* x           [[buffer(0)]],                                  \
        device const T* weight      [[buffer(1)]],                                  \
        device T*       output      [[buffer(2)]],                                  \
        constant uint&  conv_dim    [[buffer(3)]],                                  \
        constant uint&  seq_len     [[buffer(4)]],                                  \
        constant uint&  kernel_size [[buffer(5)]],                                  \
        uint3 gid [[thread_position_in_grid]]);

INST_CONV1D_FULL(float,  float)
INST_CONV1D_FULL(half,   half)
#if __METAL_VERSION__ >= 310
INST_CONV1D_FULL(bfloat, bfloat)
#endif

// ============================================================================
// Kernel 3b: save_conv_state
//
// Saves the last kernel_size positions of x into conv_state.
// Grid: (ceil(conv_dim/BT), batch_size)   Threadgroup: (BT,)
//
// x          : [B, conv_dim, S]
// conv_state : [B, conv_dim, kernel_size]
// ============================================================================

template <typename T>
kernel void gdn_save_conv_state_impl(
    device const T* x           [[buffer(0)]],
    device T*       conv_state  [[buffer(1)]],
    constant uint&  conv_dim    [[buffer(2)]],
    constant uint&  seq_len     [[buffer(3)]],
    constant uint&  kernel_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint ch = gid.x;
    const uint b  = gid.y;
    if (ch >= conv_dim) return;

    device const T* x_bch = x + (b * conv_dim + ch) * seq_len;
    device T*       cs    = conv_state + (b * conv_dim + ch) * kernel_size;

    // Save last kernel_size positions; zero-pad if seq_len < kernel_size
    for (uint i = 0; i < kernel_size; i++) {
        int src = (int)seq_len - (int)kernel_size + (int)i;
        cs[i] = (src >= 0) ? x_bch[(uint)src] : T(0.0f);
    }
}

#define INST_SAVE_CONV(T, suffix)                                                   \
    template [[host_name("gdn_save_conv_state_" #suffix)]] [[kernel]] void          \
    gdn_save_conv_state_impl<T>(                                                    \
        device const T* x           [[buffer(0)]],                                  \
        device T*       conv_state  [[buffer(1)]],                                  \
        constant uint&  conv_dim    [[buffer(2)]],                                  \
        constant uint&  seq_len     [[buffer(3)]],                                  \
        constant uint&  kernel_size [[buffer(4)]],                                  \
        uint2 gid [[thread_position_in_grid]]);

INST_SAVE_CONV(float,  float)
INST_SAVE_CONV(half,   half)
#if __METAL_VERSION__ >= 310
INST_SAVE_CONV(bfloat, bfloat)
#endif

// ============================================================================
// Kernel 4: fused_gdn_gating
//
// Fused: beta = sigmoid(b),  g = -exp(a_log) * softplus(a + dt_bias)
//
// b, a     : [total_elements]  in T (f16/bf16)
// a_log,
// dt_bias  : [num_heads]       in f32
// beta_out,
// g_out    : [total_elements]  in T
//
// Grid: (ceil(total/BT),)   Threadgroup: (BT,)
// Elements are laid out [..., num_heads] so head_idx = gid % num_heads.
//
// NOTE: total_elems guard is mandatory – during decode the dispatch may
// launch 256 threads for a buffer of only num_v_heads elements (e.g. 8).
// Without the guard, out-of-bounds threads corrupt adjacent GPU memory.
// ============================================================================

template <typename T>
kernel void gdn_gating_impl(
    device const T*     b           [[buffer(0)]],
    device const T*     a           [[buffer(1)]],
    device const float* a_log       [[buffer(2)]],
    device const float* dt_bias     [[buffer(3)]],
    device T*           beta_out    [[buffer(4)]],
    device T*           g_out       [[buffer(5)]],
    constant uint&      num_heads   [[buffer(6)]],
    constant uint&      total_elems [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    // Bounds guard – critical for decode steps where total_elems < threadgroup size.
    if (gid >= total_elems) return;

    uint head_idx = gid % num_heads;

    float b_val = float(b[gid]);
    float beta  = 1.0f / (1.0f + exp(-b_val));

    float a_val    = float(a[gid]);
    float sp_in    = a_val + dt_bias[head_idx];
    float softplus = log(1.0f + exp(sp_in));
    float g_val    = -exp(a_log[head_idx]) * softplus;

    beta_out[gid] = T(beta);
    g_out[gid]    = T(g_val);
}

#define INST_GATING(T, suffix)                                                      \
    template [[host_name("gdn_gating_" #suffix)]] [[kernel]] void                  \
    gdn_gating_impl<T>(                                                             \
        device const T*     b           [[buffer(0)]],                              \
        device const T*     a           [[buffer(1)]],                              \
        device const float* a_log       [[buffer(2)]],                              \
        device const float* dt_bias     [[buffer(3)]],                              \
        device T*           beta_out    [[buffer(4)]],                              \
        device T*           g_out       [[buffer(5)]],                              \
        constant uint&      num_heads   [[buffer(6)]],                              \
        constant uint&      total_elems [[buffer(7)]],                              \
        uint gid [[thread_position_in_grid]]);

INST_GATING(half, half)
#if __METAL_VERSION__ >= 310
INST_GATING(bfloat, bfloat)
#endif

// ============================================================================
// Kernel 5: gdn_l2_norm
//
// Fused L2-normalisation along the last dimension.
// Replaces the Candle chain: sqr → sum_keepdim → broadcast_add(eps) → sqrt
//                           → recip → broadcast_mul  (6-8 separate kernels)
// with a single cooperative kernel.
//
// Layout: [N, D]  where each row (length D) is one vector to normalise.
// All elements of one vector are handled by a single threadgroup.
//
// Grid        : (N, 1, 1)           one group per vector
// Threadgroup : (TPG, 1, 1)         TPG = min(next_power_of_2(D), 64)
// Shared mem  : (D + TPG) * float   vals[0..D) + sq_buf[0..TPG)
// ============================================================================
template <typename T>
kernel void gdn_l2_norm_impl(
    device const T*    x      [[buffer(0)]],
    device       T*    y      [[buffer(1)]],
    constant uint&     N      [[buffer(2)]],  // number of vectors
    constant uint&     D      [[buffer(3)]],  // vector length (head_dim)
    constant float&    eps    [[buffer(4)]],
    threadgroup float* shared [[threadgroup(0)]],  // (D + TPG) floats
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    if (gid >= N) return;

    threadgroup float* vals   = shared;       // [0..D)
    threadgroup float* sq_buf = shared + D;   // [D..D+tpg)

    // Phase 1: load elements into threadgroup; accumulate partial ||x||^2
    float partial_sq = 0.0f;
    for (uint i = tid; i < D; i += tpg) {
        float v = float(x[gid * D + i]);
        vals[i] = v;
        partial_sq += v * v;
    }
    sq_buf[tid] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: tree-reduction of partial sums
    for (uint s = tpg >> 1; s > 0; s >>= 1) {
        if (tid < s) sq_buf[tid] += sq_buf[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 3: compute inv-norm and write normalised values
    float inv_norm = rsqrt(sq_buf[0] + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < D; i += tpg) {
        y[gid * D + i] = T(vals[i] * inv_norm);
    }
}

#define INST_L2NORM(T, suffix)                                                      \
    template [[host_name("gdn_l2_norm_" #suffix)]] [[kernel]] void                  \
    gdn_l2_norm_impl<T>(                                                            \
        device const T*    x      [[buffer(0)]],                                    \
        device       T*    y      [[buffer(1)]],                                    \
        constant uint&     N      [[buffer(2)]],                                    \
        constant uint&     D      [[buffer(3)]],                                    \
        constant float&    eps    [[buffer(4)]],                                    \
        threadgroup float* shared [[threadgroup(0)]],                               \
        uint gid [[threadgroup_position_in_grid]],                                  \
        uint tid [[thread_position_in_threadgroup]],                                \
        uint tpg [[threads_per_threadgroup]]);

INST_L2NORM(float,  float)
INST_L2NORM(half,   half)
#if __METAL_VERSION__ >= 310
INST_L2NORM(bfloat, bfloat)
#endif

// ============================================================================
// Kernel 6: gdn_rms_norm_gated
//
// Fused RMSNorm + SiLU-gate + weight scaling for the GDN output normalisation.
// Replaces the 12-kernel Candle chain in RmsNormGated::forward:
//   x→F32, gate→F32, silu(gate), sqr(x), mean_keepdim, +eps, sqrt,
//   broadcast_div, weight→F32, broadcast_mul(weight), broadcast_mul(gate), →dtype
//
// Computes:  out[i] = rms_norm(x[i]) * weight[i] * silu(gate[i])
//   where rms_norm(x)[i] = x[i] / sqrt(mean(x^2) + eps)
//
// x, gate : [N, D]  in src dtype (F16 or BF16)
// weight  : [D]     in F32
// out     : [N, D]  in src dtype
//
// Grid        : (N, 1, 1)
// Threadgroup : (TPG, 1, 1)   TPG = min(next_power_of_2(D), 64)
// Shared mem  : (D + TPG) * float
// ============================================================================
template <typename T>
kernel void gdn_rms_norm_gated_impl(
    device const T*     x      [[buffer(0)]],
    device const T*     gate   [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device       T*     out    [[buffer(3)]],
    constant uint&      N      [[buffer(4)]],
    constant uint&      D      [[buffer(5)]],
    constant float&     eps    [[buffer(6)]],
    threadgroup float*  shared [[threadgroup(0)]],  // (D + TPG) floats
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    if (gid >= N) return;

    threadgroup float* vals   = shared;      // x in F32 [D]
    threadgroup float* sq_buf = shared + D;  // partial sq sums [tpg]

    // Phase 1: load x[gid, :] into threadgroup, accumulate partial ||x||^2
    float partial_sq = 0.0f;
    for (uint i = tid; i < D; i += tpg) {
        float v = float(x[gid * D + i]);
        vals[i] = v;
        partial_sq += v * v;
    }
    sq_buf[tid] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: tree-reduce sq sums → mean sq
    for (uint s = tpg >> 1; s > 0; s >>= 1) {
        if (tid < s) sq_buf[tid] += sq_buf[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(sq_buf[0] / float(D) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: normalize, scale by weight, multiply by silu(gate)
    for (uint i = tid; i < D; i += tpg) {
        float g    = float(gate[gid * D + i]);
        float silu = g / (1.0f + exp(-g));   // silu(x) = x * sigmoid(x)
        float v    = vals[i] * inv_rms * weight[i] * silu;
        out[gid * D + i] = T(v);
    }
}

#define INST_RMS_NORM_GATED(T, suffix)                                              \
    template [[host_name("gdn_rms_norm_gated_" #suffix)]] [[kernel]] void           \
    gdn_rms_norm_gated_impl<T>(                                                     \
        device const T*     x      [[buffer(0)]],                                   \
        device const T*     gate   [[buffer(1)]],                                   \
        device const float* weight [[buffer(2)]],                                   \
        device       T*     out    [[buffer(3)]],                                   \
        constant uint&      N      [[buffer(4)]],                                   \
        constant uint&      D      [[buffer(5)]],                                   \
        constant float&     eps    [[buffer(6)]],                                   \
        threadgroup float*  shared [[threadgroup(0)]],                              \
        uint gid [[threadgroup_position_in_grid]],                                  \
        uint tid [[thread_position_in_threadgroup]],                                \
        uint tpg [[threads_per_threadgroup]]);

INST_RMS_NORM_GATED(float,  float)
INST_RMS_NORM_GATED(half,   half)
#if __METAL_VERSION__ >= 310
INST_RMS_NORM_GATED(bfloat, bfloat)
#endif

// ============================================================================
// Kernel 7: gdn_gating_strided
//
// Same computation as gdn_gating but reads b and a directly from the fused
// projection output buffer (layout [B*S, total_col]) using column offsets,
// eliminating the two contiguous-copy dispatches that gdn_gating requires
// when b and a are narrow (non-contiguous) views of the projection output.
//
// proj       : [B*S, total_col]  in InT — the raw matmul output, contiguous.
// b_col      : column index where b values start (= z_end in FusedAll)
// a_col      : column index where a values start (= b_end in FusedAll)
// total_col  : number of columns in proj (= total_out_dim)
// a_log      : [num_heads]  in F32
// dt_bias    : [num_heads]  in F32
// beta_out,
// g_out      : [total_elems]  in InT   (total_elems = B * S * num_heads)
//
// Grid: (ceil(total_elems / BT),)   Threadgroup: (BT,)
// ============================================================================
template <typename T>
kernel void gdn_gating_strided_impl(
    device const T*     proj        [[buffer(0)]],
    device const float* a_log       [[buffer(1)]],
    device const float* dt_bias     [[buffer(2)]],
    device T*           beta_out    [[buffer(3)]],
    device T*           g_out       [[buffer(4)]],
    constant uint&      num_heads   [[buffer(5)]],
    constant uint&      total_col   [[buffer(6)]],
    constant uint&      b_col       [[buffer(7)]],
    constant uint&      a_col       [[buffer(8)]],
    constant uint&      total_elems [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total_elems) return;

    uint head_idx = gid % num_heads;
    uint row_idx  = gid / num_heads;  // B*S flat index

    float b_val = float(proj[row_idx * total_col + b_col + head_idx]);
    float a_val = float(proj[row_idx * total_col + a_col + head_idx]);

    float beta = 1.0f / (1.0f + exp(-b_val));

    float sp_in    = a_val + dt_bias[head_idx];
    float softplus = log(1.0f + exp(sp_in));
    float g_val    = -exp(a_log[head_idx]) * softplus;

    beta_out[gid] = T(beta);
    g_out[gid]    = T(g_val);
}

#define INST_GATING_STRIDED(T, suffix)                                              \
    template [[host_name("gdn_gating_strided_" #suffix)]] [[kernel]] void           \
    gdn_gating_strided_impl<T>(                                                     \
        device const T*     proj        [[buffer(0)]],                              \
        device const float* a_log       [[buffer(1)]],                              \
        device const float* dt_bias     [[buffer(2)]],                              \
        device T*           beta_out    [[buffer(3)]],                              \
        device T*           g_out       [[buffer(4)]],                              \
        constant uint&      num_heads   [[buffer(5)]],                              \
        constant uint&      total_col   [[buffer(6)]],                              \
        constant uint&      b_col       [[buffer(7)]],                              \
        constant uint&      a_col       [[buffer(8)]],                              \
        constant uint&      total_elems [[buffer(9)]],                              \
        uint gid [[thread_position_in_grid]]);

INST_GATING_STRIDED(half, half)
#if __METAL_VERSION__ >= 310
INST_GATING_STRIDED(bfloat, bfloat)
#endif

// ============================================================================
// Kernel 8: gdn_l2_norm2
//
// Fused L2-normalisation of two tensors (q and k) in a single dispatch.
// Replaces two calls to gdn_l2_norm with one, saving one Metal command-buffer
// submission per GDN layer per decode/prefill step.
//
// Both xq and xk must have the same shape [..., D] and be contiguous.
// Grid: (2*N, 1, 1) — groups [0, N) normalise q; groups [N, 2N) normalise k.
// Each threadgroup works independently on one vector.
//
// xq, xk   : [N, D]  in T
// yq, yk   : [N, D]  in T  (output)
// N, D, eps: same as gdn_l2_norm
// ============================================================================
template <typename T>
kernel void gdn_l2_norm2_impl(
    device const T*    xq     [[buffer(0)]],
    device       T*    yq     [[buffer(1)]],
    device const T*    xk     [[buffer(2)]],
    device       T*    yk     [[buffer(3)]],
    constant uint&     N      [[buffer(4)]],
    constant uint&     D      [[buffer(5)]],
    constant float&    eps    [[buffer(6)]],
    threadgroup float* shared [[threadgroup(0)]],  // (D + TPG) floats
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    if (gid >= 2 * N) return;

    // Select which tensor this group normalises.
    bool is_k     = (gid >= N);
    uint vec_idx  = is_k ? (gid - N) : gid;

    device const T* x = is_k ? xk : xq;
    device       T* y = is_k ? yk : yq;

    threadgroup float* vals   = shared;       // [0..D)
    threadgroup float* sq_buf = shared + D;   // [D..D+tpg)

    float partial_sq = 0.0f;
    for (uint i = tid; i < D; i += tpg) {
        float v = float(x[vec_idx * D + i]);
        vals[i] = v;
        partial_sq += v * v;
    }
    sq_buf[tid] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tpg >> 1; s > 0; s >>= 1) {
        if (tid < s) sq_buf[tid] += sq_buf[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_norm = rsqrt(sq_buf[0] + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < D; i += tpg) {
        y[vec_idx * D + i] = T(vals[i] * inv_norm);
    }
}

#define INST_L2NORM2(T, suffix)                                                     \
    template [[host_name("gdn_l2_norm2_" #suffix)]] [[kernel]] void                 \
    gdn_l2_norm2_impl<T>(                                                           \
        device const T*    xq     [[buffer(0)]],                                    \
        device       T*    yq     [[buffer(1)]],                                    \
        device const T*    xk     [[buffer(2)]],                                    \
        device       T*    yk     [[buffer(3)]],                                    \
        constant uint&     N      [[buffer(4)]],                                    \
        constant uint&     D      [[buffer(5)]],                                    \
        constant float&    eps    [[buffer(6)]],                                    \
        threadgroup float* shared [[threadgroup(0)]],                               \
        uint gid [[threadgroup_position_in_grid]],                                  \
        uint tid [[thread_position_in_threadgroup]],                                \
        uint tpg [[threads_per_threadgroup]]);

INST_L2NORM2(float,  float)
INST_L2NORM2(half,   half)
#if __METAL_VERSION__ >= 310
INST_L2NORM2(bfloat, bfloat)
#endif
