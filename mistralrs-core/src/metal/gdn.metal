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

#define GDN_BV 64u

kernel void gdn_recurrence(
    device const float* q       [[buffer(0)]],
    device const float* k       [[buffer(1)]],
    device const float* v       [[buffer(2)]],
    device const float* g       [[buffer(3)]],
    device const float* beta    [[buffer(4)]],
    device float*       state   [[buffer(5)]],
    device float*       output  [[buffer(6)]],
    constant uint&      seq_len [[buffer(7)]],
    constant uint&      k_dim   [[buffer(8)]],
    constant uint&      v_dim   [[buffer(9)]],
    threadgroup float*  shared  [[threadgroup(0)]],   // 2 * k_dim floats
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]])
{
    const uint v_tile  = tgpig.x;
    const uint bh      = tgpig.y;
    const uint tid     = tpitg.x;
    const uint v_idx   = v_tile * GDN_BV + tid;
    // NOTE: do NOT early-return here – threads beyond v_dim must still
    // participate in all threadgroup_barrier() calls to avoid deadlock.
    // They just skip reads/writes of their (invalid) column.
    const bool valid   = (v_idx < v_dim);

    const uint K = k_dim;
    const uint V = v_dim;

    device const float* q_bh    = q    + bh * seq_len * K;
    device const float* k_bh    = k    + bh * seq_len * K;
    device const float* v_bh    = v    + bh * seq_len * V;
    device const float* g_bh    = g    + bh * seq_len;
    device const float* beta_bh = beta + bh * seq_len;
    // State column for this (bh, v_idx): st_col[j] = state[bh, j, v_idx]
    // addressed as st_bh[j * V + v_idx].  Each thread owns one column.
    device float* st_bh  = state  + bh * K * V;
    device float* out_bh = output + bh * seq_len * V;

    // Shared memory layout: k_buf[K] | q_buf[K]
    threadgroup float* k_buf = shared;
    threadgroup float* q_buf = shared + K;

    for (uint t = 0; t < seq_len; t++) {
        // ---- Step A: cooperatively load k_t --------------------------------
        // ALL threads (including invalid ones) participate so every k_buf[j]
        // is filled before the barrier.
        for (uint j = tid; j < K; j += GDN_BV) {
            k_buf[j] = k_bh[t * K + j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float decay  = valid ? exp(g_bh[t])    : 1.0f;
        float beta_t = valid ? beta_bh[t]       : 0.0f;
        float v_t    = valid ? v_bh[t * V + v_idx] : 0.0f;

        // ---- Pass 1: decay state, accumulate kv_mem ------------------------
        // Each valid thread owns its own column (j * V + v_idx); invalid
        // threads skip all device-memory accesses for this column.
        float kv_mem = 0.0f;
        if (valid) {
            for (uint j = 0; j < K; j++) {
                float s_j = st_bh[j * V + v_idx] * decay;
                st_bh[j * V + v_idx] = s_j;
                kv_mem = fma(s_j, k_buf[j], kv_mem);
            }
        }

        float delta = (v_t - kv_mem) * beta_t;

        // ---- Step B: cooperatively load q_t --------------------------------
        for (uint j = tid; j < K; j += GDN_BV) {
            q_buf[j] = q_bh[t * K + j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Pass 2: update state, accumulate output -----------------------
        float y_t = 0.0f;
        if (valid) {
            for (uint j = 0; j < K; j++) {
                float s_j = fma(k_buf[j], delta, st_bh[j * V + v_idx]);
                st_bh[j * V + v_idx] = s_j;
                y_t = fma(s_j, q_buf[j], y_t);
            }
            out_bh[t * V + v_idx] = y_t;
        }

        // Synchronize before next iteration overwrites k_buf / q_buf.
        // ALL threads reach this barrier (invalid ones execute the no-op
        // branches above and arrive here at the same time as valid threads).
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // State is already fully written back to device memory in the loops above.
}

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

template <typename T>
kernel void gdn_conv1d_update_impl(
    device const T* x           [[buffer(0)]],
    device const T* weight      [[buffer(1)]],
    device T*       conv_state  [[buffer(2)]],
    device T*       output      [[buffer(3)]],
    constant uint&  conv_dim    [[buffer(4)]],
    constant uint&  kernel_size [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint ch = gid.x;
    const uint b  = gid.y;
    if (ch >= conv_dim) return;

    device T*       cs = conv_state + (b * conv_dim + ch) * kernel_size;
    device const T* w  = weight + ch * kernel_size;

    // Shift ring buffer left by 1, insert new token value at the end
    for (uint i = 0; i < kernel_size - 1; i++) {
        cs[i] = cs[i + 1];
    }
    cs[kernel_size - 1] = x[b * conv_dim + ch];

    // Dot product with weight, then SiLU
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
        device const T* x           [[buffer(0)]],                                  \
        device const T* weight      [[buffer(1)]],                                  \
        device T*       conv_state  [[buffer(2)]],                                  \
        device T*       output      [[buffer(3)]],                                  \
        constant uint&  conv_dim    [[buffer(4)]],                                  \
        constant uint&  kernel_size [[buffer(5)]],                                  \
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
