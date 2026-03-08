/// gemv.metal — Fast matrix-vector multiply for decode (single-token) steps.
///
/// All linear layers become GEMVs when batch_size=1, seq_len=1. Candle routes
/// these through MLX GEMM with 32×32 tiles, wasting 31 of 32 output rows.
/// These kernels are explicit GEMV implementations that fully saturate memory
/// bandwidth for the one output row we actually need.
///
/// Layout convention (matches QuantMethod::forward / matmul convention):
///   weights W  : [N_out, N_in]  (transposed — row = one output neuron)
///   input  x   : [M, N_in]      (M = batch_size × seq_len, typically 1)
///   output y   : [M, N_out]
///   y[m, n]    = dot(x[m, :], W[n, :])
///
/// Kernel grid and threadgroup sizing:
///   grid   : (M, ceil(N_out / TG_N))
///   threads: (TG_K, TG_N)
///   — TG_K threads cooperate on the K-reduction per output element.
///   — TG_N output elements are computed per threadgroup (one per "column").
///   — Shared memory holds TG_K × TG_N partial sums.

#include <metal_stdlib>
using namespace metal;

// ── Tuning constants ──────────────────────────────────────────────────────
// TG_K:  number of threads working on the K (inner) reduction per output row.
//        Must be a power of 2.  32 (one simdgroup) is a good default.
// TG_N:  number of output elements computed per threadgroup.
//        Product TG_K * TG_N must be ≤ 1024 (Metal thread limit).
#define GEMV_TG_K 32
#define GEMV_TG_N  4

// ── Scalar GEMV template ─────────────────────────────────────────────────
//
// WType: weight dtype (half / bfloat / float)
// XType: input  dtype (same as WType for native-precision inference)
//
// The inner accumulation is always float32 for numerical stability.
template <typename WType, typename XType>
kernel void gemv_impl(
    device const WType*  weights [[buffer(0)]],   // [N_out, N_in]
    device const XType*  input   [[buffer(1)]],   // [M, N_in]
    device       WType*  output  [[buffer(2)]],   // [M, N_out]
    constant uint& N_out         [[buffer(3)]],
    constant uint& N_in          [[buffer(4)]],
    constant uint& M             [[buffer(5)]],   // batch rows (usually 1)
    threadgroup float* shared    [[threadgroup(0)]], // GEMV_TG_K * GEMV_TG_N floats
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]])
{
    const uint m_row   = tgpig.x;
    const uint n_start = tgpig.y * GEMV_TG_N;

    const uint k_tid = tpitg.x;
    const uint n_tid = tpitg.y;

    const uint n_abs = n_start + n_tid;
    const bool valid_n = (n_abs < N_out);
    const bool valid_m = (m_row < M);

    device const WType* w_row = weights + n_abs * N_in;
    device const XType* x_row = input   + m_row * N_in;

    float partial = 0.0f;
    if (valid_n && valid_m) {
        for (uint k = k_tid; k < N_in; k += GEMV_TG_K) {
            partial = fma(float(w_row[k]), float(x_row[k]), partial);
        }
    }

    shared[n_tid * GEMV_TG_K + k_tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = GEMV_TG_K / 2; stride > 0; stride >>= 1) {
        if (k_tid < stride) {
            shared[n_tid * GEMV_TG_K + k_tid] +=
                shared[n_tid * GEMV_TG_K + k_tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (k_tid == 0 && valid_n && valid_m) {
        output[m_row * N_out + n_abs] = WType(shared[n_tid * GEMV_TG_K]);
    }
}

#define INST_GEMV(WT, XT, suffix)                                               \
    template [[host_name("gemv_" #suffix)]] [[kernel]] void gemv_impl<WT, XT>( \
        device const WT* weights  [[buffer(0)]],                                \
        device const XT* input    [[buffer(1)]],                                \
        device       WT* output   [[buffer(2)]],                                \
        constant uint& N_out      [[buffer(3)]],                                \
        constant uint& N_in       [[buffer(4)]],                                \
        constant uint& M          [[buffer(5)]],                                \
        threadgroup float* shared [[threadgroup(0)]],                           \
        uint3 tgpig [[threadgroup_position_in_grid]],                           \
        uint3 tpitg [[thread_position_in_threadgroup]]);

INST_GEMV(float,  float,  f32)
INST_GEMV(half,   half,   f16)
#if __METAL_VERSION__ >= 310
INST_GEMV(bfloat, bfloat, bf16)
#endif
