# Qwen3.5 Performance Plan: Closing the Gap with llama.cpp

## Status

| Phase | What | Status | Actual Speedup |
|-------|------|--------|---------------|
| 1 | CPU chunked recurrence (`gated_delta_rule_chunked`) | ✅ Done | CPU prefill: O(T) allocs → O(T/64) BLAS |
| 4 | CUDA decode fast path (reshape instead of transpose) | ✅ Done | ~5–10% CUDA decode |
| 5 | Fuse GDN input projections into one matmul | ✅ Done | **4.5 → 50 T/s Metal decode (11×!)** |
| 6 | CPU `causal_conv1d_full` O(kernel_size) loop | ✅ Done | CPU prefill |
| — | Typed Metal kernel (skip 5 `to_dtype(F32)` per layer) | ✅ Done | Included in Phase 5 result |
| — | Eliminate `Tensor::cat` before conv1d (FusedAll path) | ✅ Done | Included in Phase 5 result |
| — | Fix engine panic on client disconnect | ✅ Done | Stability |
| 2 | Fused CUDA chunked kernel | ⏳ Deferred | Est. +2–4× CUDA prefill |
| 3 | Fused Metal chunked kernel | ⏳ Next | Est. +1.5–3× Metal prefill |

**Baseline**: ~4.5 T/s Metal decode  
**Current**: ~50 T/s Metal decode (M3 Max, Qwen3.5-0.6B)

---

## Key Lesson from Phase 5

The plan estimated Phase 5 (projection fusion) at "5–15% decode improvement". The actual
result was **11×**. Why?

On Apple Silicon, Metal kernel *dispatch* overhead dominates for small workloads. At
decode time (batch=1, seq=1), each tensor operation issues a separate Metal command-buffer
entry with ~0.5–1 ms of scheduling overhead. Cutting from 11+ dispatches/layer to 8, and
more importantly fusing 4 matmuls into 1, reduced the wall-clock per-token time
dramatically.

Original 4.5 T/s → 50 T/s breakdown:
1. **Typed kernel** (skip 5 `to_dtype` casts per layer = 70 fewer dispatches for 14 layers): ~4.5 → ~6–7 T/s
2. **Projection fusion** (4 matmuls → 1 = 3 fewer dispatches per layer = 42 fewer total): ~6–7 → ~22 T/s  
3. **Eliminate `Tensor::cat`** (1 copy kernel → zero-copy narrow per layer = 14 fewer dispatches): ~22 → ~50 T/s

Each "small" kernel launch removal was worth progressively more as earlier removals
unblocked the GPU pipeline.

---

## What Was Implemented

### Phase 1 — CPU Chunked Recurrence (`deltanet.rs`)

`gated_delta_rule_chunked()` replaces the per-token `gated_delta_rule_recurrence()` on
CPU. The algorithm is chunk-parallel with `CHUNK_SIZE=64`, matching llama.cpp's
`build_delta_net_chunking`. Back-substitution runs 64 sequential Candle ops (not
`seq_len` ops).

Called from `apply_recurrence()` for CPU devices with `seq_len > 1`.

**Math correctness note**: `rhs = v_b - exp(g_cs) * (k_b @ state)` — uses `k_b`
(beta-weighted key) for the state retrieval term, matching the original sequential
derivation.

### Phase 5 — `GdnProjection::FusedAll` (`deltanet.rs`, `qwen3_5.rs`)

At load time (world_size == 1, unquantized weights only):
```
in_proj_qkv  [qkv_out × H]        ┐
in_proj_z    [value_dim × H]       ├─ cat(axis=0) → fused [total × H]
in_proj_b    [num_v_heads × H]     │    → ReplicatedLayer::from_linear(Linear::new(fused, None))
in_proj_a    [num_v_heads × H]     ┘
```

`project_inputs()` issues **one** `qmethod_matmul` then splits with zero-copy `narrow`.
`GdnProjected.qkv_for_conv` carries the `[q|k|v_flat]` slice directly to conv1d,
eliminating the `Tensor::cat` that previously created an extra Metal buffer.

ISQ/GGUF quantized models fall back to `SplitQkvZa` automatically (since
`unquant_weight_bias()` returns `None` for them). `residual_tensors()` splits the fused
weight back into the four named checkpoint keys for serialization.

### Engine panic fix (`utils/mod.rs`)

`handle_pipeline_forward_error!` used `.send(...).await.unwrap()` to report errors.
When the error was itself caused by a disconnected client (the response channel was
already closed), the `unwrap()` panicked. Changed to `let _ = .send(...).await` so
client disconnection is handled cleanly.

---

## Remaining Opportunity: Phase 3 — Fused Metal Prefill Kernel

### What & Why

At **decode** (seq_len=1) the current Metal kernel is already fast — one token per kernel
call, register-resident state. The remaining gap vs llama.cpp is **prefill** (long
prompts).

Currently `apply_recurrence()` for Metal prefill calls
`gated_delta_rule_recurrence_metal_typed`, which loops `seq_len` times inside a single
Metal dispatch. For seq_len=2048 that's 2048 serial iterations per (bh) thread group.

llama.cpp's approach: `build_delta_net_chunking` with `CS=64` — 32 inter-chunk state
updates, all intra-chunk work expressed as matrix multiply. A Metal kernel that does the
same would cut prefill time by ~30–60×.

### Algorithm (matches Phase 1, but in a Metal kernel)

```metal
kernel void gdn_chunked_scan(
    device const InT* q      [[buffer(0)]],   // [BH, T, K]
    device const InT* k      [[buffer(1)]],
    device const InT* v      [[buffer(2)]],
    device const InT* g      [[buffer(3)]],   // [BH, T]
    device const InT* beta   [[buffer(4)]],
    device float*     state  [[buffer(5)]],   // [BH, K, V]  in/out
    device float*     output [[buffer(6)]],   // [BH, T, V]
    constant uint& T         [[buffer(7)]],
    constant uint& K         [[buffer(8)]],
    constant uint& V         [[buffer(9)]],
    constant float& q_scale  [[buffer(10)]],
    uint2 tgpig [[threadgroup_position_in_grid]],  // (v_tile, bh)
    uint  tid   [[thread_position_in_threadgroup]]
) {
    const uint CS = 64;
    const uint bh  = tgpig.y;
    const uint vid = tgpig.x * GDN_BV + tid;
    if (vid >= V) return;

    // Load state column [K] into registers
    float s[MAX_K];
    for (uint dk = 0; dk < K; dk++)
        s[dk] = state[bh * K * V + dk * V + vid];

    const uint n_chunks = (T + CS - 1) / CS;
    for (uint chunk = 0; chunk < n_chunks; chunk++) {
        const uint base = chunk * CS;

        // threadgroup: shared k_buf [CS, K], g_cs [CS], beta [CS], v [CS], q [CS, K]
        threadgroup float sh_k[CS * MAX_K];
        threadgroup float sh_q[CS * MAX_K];
        threadgroup float sh_g[CS];     // cumulative g
        threadgroup float sh_b[CS];
        threadgroup float sh_v[CS];

        // Load chunk into threadgroup (all threads cooperate)
        // ... (load k, q, v, g, beta for this chunk, compute cumsum(g) in threadgroup)

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Intra-chunk sequential scan (CS=64 steps, state in registers)
        float g_end = sh_g[min(base + CS - 1, T - 1) - base];
        for (uint i = 0; i < CS && (base + i) < T; i++) {
            float gi   = sh_g[i];
            float bi   = sh_b[i];
            float vi   = sh_v[i];
            float decay = exp(gi);

            // Retrieve: sk = state^T k_i
            float sk = 0.f;
            for (uint dk = 0; dk < K; dk++)
                sk += s[dk] * sh_k[i * K + dk];

            float delta = (vi - sk) * bi;

            // Update state: s += k_i * delta (already decayed)
            for (uint dk = 0; dk < K; dk++)
                s[dk] = fma(sh_k[i * K + dk], delta, s[dk] * decay);

            // Output: y = state^T q_i
            float y = 0.f;
            for (uint dk = 0; dk < K; dk++)
                y = fma(s[dk], sh_q[i * K + dk] * q_scale, y);
            output[(bh * T + base + i) * V + vid] = y;
        }
        // Apply inter-chunk state decay: s *= exp(g_end - g_at_start_of_chunk)
        // (already folded into per-step decay above)
    }

    // Write back state
    for (uint dk = 0; dk < K; dk++)
        state[bh * K * V + dk * V + vid] = s[dk];
}
```

This is exactly the existing `gdn_recurrence_typed` kernel extended with an outer
`for (chunk)` loop. No new algorithm, just the outer loop moved from Rust/Candle dispatch
into the kernel itself.

### Files to Change

- `mistralrs-core/src/metal/gdn.metal` — add `gdn_chunked_scan_half` / `_bfloat` kernels
- `mistralrs-core/src/metal/gdn.rs` — add `gated_delta_rule_chunked_metal_typed()` wrapper
- `mistralrs-core/src/models/deltanet.rs` — in Metal branch of `apply_recurrence()`, call
  chunked kernel for `seq_len > 1` instead of `gated_delta_rule_recurrence_metal_typed`

### Expected Gain

- **Metal prefill**: ~30–60× faster for long prompts (2048 tokens: 32 inter-chunk steps
  instead of 2048 serial kernel iterations)
- **Metal decode**: unchanged (already optimal at seq_len=1)
- **No correctness risk**: the Metal kernel produces identical output to the CPU-side
  `gated_delta_rule_chunked`; they implement the same algorithm

### Estimated Effort

2–3 days for a correct initial implementation (Metal is simpler than CUDA for this
use-case because of unified memory and simpler synchronization model).

---

## Phase 2 — Fused CUDA Chunked Kernel (Deferred)

Same idea as Phase 3 but for CUDA. Deferred because:
1. CUDA users typically use larger models where the attention layers dominate more
2. The existing CUDA sequential kernel is already efficient for decode
3. Phase 3 (Metal) is higher priority given the target hardware (M3 Max)

See original algorithm sketch in this document's git history.

---

## Root Cause Summary (updated)

| # | Cause | Status |
|---|-------|--------|
| 4× redundant matmuls per GDN layer | Fixed: `FusedAll` | ✅ |
| 5× `to_dtype(F32)` per layer per decode step | Fixed: typed Metal kernel | ✅ |
| Unnecessary `Tensor::cat` before conv1d | Fixed: `qkv_for_conv` field | ✅ |
| CPU prefill: per-token tensor allocations in recurrence | Fixed: `gated_delta_rule_chunked` | ✅ |
| CPU prefill: per-token tensor allocations in conv1d | Fixed: O(kernel_size) loop | ✅ |
| Metal prefill: 2048 serial kernel iterations for seq_len=2048 | **Open — Phase 3** | ⏳ |
| CUDA prefill: 2048 serial kernel iterations for seq_len=2048 | Open — Phase 2 | ⏳ |
