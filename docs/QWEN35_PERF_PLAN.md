# Qwen3.5 Performance Plan: Closing the Gap with llama.cpp

## Root Cause Summary

Qwen3.5 uses hybrid layers: standard full-attention interleaved with GatedDeltaNet (GDN)
linear-attention. The GDN layers are the bottleneck. The performance gap vs llama.cpp has
four concrete causes, ranked by impact:

| # | Cause | Affects | Est. Speedup |
|---|-------|---------|--------------|
| 1 | No chunked prefill — CUDA/Metal kernels loop token-by-token over full `seq_len` | Prefill (long prompts) | 10–40× on prefill |
| 2 | Sequential Rust `for` loop fallback on CPU/Metal/CUDA — each step allocates tensors | CPU, any device without our fused kernel | 5–15× on prefill |
| 3 | ~130 `.contiguous()` / `.transpose()` calls in `deltanet.rs` forward, many launch individual GPU copy kernels | Decode, all devices | 10–25% on decode |
| 4 | 4–6 separate input projections for GDN (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`) — each is its own matmul | Decode, all devices | 5–15% on decode |

llama.cpp's key advantages:
- **Chunked scan** (`build_delta_net_chunking`, CS=64): replaces `seq_len` sequential steps
  with `seq_len/64` inter-chunk state updates; all intra-chunk work is expressed as GEMM.
- **Static compute graph**: GGML schedules and fuses ops across the entire forward pass.
- **`ggml_gated_delta_net`**: single fused op for autoregressive step.

---

## Phase 1 — Chunked GDN Prefill via Candle Tensor Ops *(highest ROI, device-agnostic)*

### What & Why

Replace the current device-specific sequential kernels with a chunk-parallel Candle
implementation that works on **all backends** (CPU, CUDA, Metal). The algorithm mirrors
`build_delta_net_chunking` in `llama.cpp/src/models/delta-net-base.cpp`.

Chunk size `CS = 64`. Instead of `seq_len` serial state-update steps, we do
`n_chunks = ceil(seq_len / CS)` serial steps; within each chunk everything is batched GEMM.

### Files to Change

- `mistralrs-core/src/models/deltanet.rs` — add `gated_delta_rule_chunked()`, call it from
  `apply_recurrence()` when `seq_len > 1`

### Algorithm (matches llama.cpp's `build_delta_net_chunking`)

```
// Input tensors already in shape [B, T, H, D] (batch, seq, heads, dim)
// state: [B, H, D_k, D_v]  (recurrent state, carried across chunks)

fn gated_delta_rule_chunked(q, k, v, g, beta, state, chunk_size=64) -> (Tensor, Tensor) {
    let (B, T, H, Dk) = q.dims4();
    let Dv = v.dim(D::Minus1);
    let n_chunks = T.div_ceil(chunk_size);
    let pad = n_chunks * chunk_size - T;
    let mut outputs = Vec::with_capacity(n_chunks);

    // --- Pad to multiple of chunk_size if necessary ---
    let q = pad_seq(q, pad);       // [B, n_chunks*CS, H, Dk]
    let k = pad_seq(k, pad);
    let v = pad_seq(v, pad);
    let g = pad_seq(g, pad);       // [B, n_chunks*CS, H]
    let beta = pad_seq(beta, pad); // [B, n_chunks*CS, H]

    for chunk in 0..n_chunks {
        let sl = chunk * CS .. (chunk + 1) * CS;

        // Slice chunk [B, CS, H, D]
        let q_c    = q.narrow(1, sl.start, CS);
        let k_c    = k.narrow(1, sl.start, CS);
        let v_c    = v.narrow(1, sl.start, CS);
        let g_c    = g.narrow(1, sl.start, CS);
        let beta_c = beta.narrow(1, sl.start, CS);

        // 1. Intra-chunk cumulative sum of log-decay g
        //    g_cs[b, i, h] = sum_{t<=i} g_c[b, t, h]    shape [B, CS, H]
        let g_cs = g_c.cumsum(1);

        // 2. Causal decay mask  [B, H, CS, CS]
        //    decay[b, h, i, j] = exp(g_cs[j] - g_cs[i])  if j <= i, else 0
        let g_i = g_cs.unsqueeze(-1);          // [B, CS, H, 1]
        let g_j = g_cs.unsqueeze(-2);          // [B, 1, H, CS]  (broadcast as j dimension)
        let raw  = (g_j - g_i).exp();          // [B, CS, H, CS]  -- j rows, i cols
        let mask = lower_triangular_mask(CS);  // [CS, CS]  (1 on and below diagonal)
        let decay = raw * mask;                // [B, CS, H, CS]  -- causal

        // 3. Beta-weighted k and v
        let k_b = k_c * beta_c.unsqueeze(-1);  // [B, CS, H, Dk]
        let v_b = v_c * beta_c.unsqueeze(-1);  // [B, CS, H, Dv]

        // 4. Intra-chunk key-key and key-query attention matrices via batched GEMM
        //    Reshape to fuse B and H: [B*H, CS, D]
        let q_bh  = q_c .permute([0,2,1,3]).reshape([B*H, CS, Dk]);
        let k_bh  = k_c .permute([0,2,1,3]).reshape([B*H, CS, Dk]);
        let k_b_bh= k_b .permute([0,2,1,3]).reshape([B*H, CS, Dk]);
        let v_bh  = v_c .permute([0,2,1,3]).reshape([B*H, CS, Dv]);
        let v_b_bh= v_b .permute([0,2,1,3]).reshape([B*H, CS, Dv]);
        let dec   = decay.permute([0,2,1,3]).reshape([B*H, CS, CS]);
        let scale = 1.0 / (Dk as f64).sqrt();

        //  kb[i,j] = (k_b[i] · k[j]) * decay[i,j]    [B*H, CS, CS]
        let kb = k_bh.matmul(k_b_bh.transpose(1,2)) * dec.clone();

        //  kq[i,j] = (k[i] · q[j]) * decay[i,j]      [B*H, CS, CS]
        let kq = (k_bh.matmul(q_bh.transpose(1,2)) * scale) * dec;

        // 5. Solve lower-triangular system: attn = (I - tril_strict(kb))^{-1} * tril_diag(kb)
        //    Approximated by back-substitution over CS=64 steps (cheap, not seq_len steps)
        //    or use the additive Neumann series for small spectral radius (||kb|| < 1 in practice).
        let attn = lower_tri_solve(kb);   // [B*H, CS, CS]  (see impl note below)

        // 6. Intra-chunk output:  o = v_b^T @ attn  +  state @ q_g_exp
        //    g_exp[i] = exp(g_cs[i])  [B*H, CS]
        let g_exp = g_cs.permute([0,2,1]).reshape([B*H, CS]).exp();

        //    Cross-term: state contribution to each query position
        //    o_inter[b*h, i, :] = state[b*h, :, :] @ (q[i] * g_exp[i])
        let q_g = (q_bh * g_exp.unsqueeze(-1));         // [B*H, CS, Dk]
        let o_inter = q_g.matmul(state_flat);            // [B*H, CS, Dv]

        //    Intra-chunk: o_intra[i] = sum_{j<=i} attn[i,j] * v_b[j]
        let o_intra = attn.matmul(v_b_bh);               // [B*H, CS, Dv]
        let o_chunk = o_inter + o_intra;                  // [B*H, CS, Dv]

        outputs.push(o_chunk.reshape([B, H, CS, Dv]).permute([0,2,1,3]));

        // 7. Update recurrent state (one step per chunk, not per token)
        //    g_last = exp(g_cs[:, -1, :])  [B*H, 1]  -- last cumsum in chunk
        //    key_gdiff[i] = k[i] * exp(g_cs[-1] - g_cs[i])
        //    new_state = state * g_last + sum_i(key_gdiff[i]^T @ v_b[i])
        let g_last = g_exp.narrow(1, CS-1, 1);           // [B*H, 1]
        let g_diff = (g_exp.narrow(1,CS-1,1) - g_exp).exp(); // [B*H, CS] -- decay from end
        let k_gdiff = k_bh * g_diff.unsqueeze(-1);       // [B*H, CS, Dk]
        let kv_update = k_gdiff.transpose(1,2).matmul(v_b_bh); // [B*H, Dk, Dv]
        state_flat = state_flat * g_last.unsqueeze(-1) + kv_update;
    }

    // Trim padding from output
    let out = Tensor::cat(&outputs, 1).narrow(1, 0, T);
    (out, state_flat.reshape([B, H, Dk, Dv]))
}
```

**`lower_tri_solve` implementation note**: For CS=64, implement a plain
Rust-level loop over 64 iterations (not `seq_len`). This is essentially free
compared to a full-sequence sequential recurrence:

```rust
fn lower_tri_solve(kb: Tensor) -> Result<Tensor> {
    // kb: [BH, CS, CS], strict lower-triangular part gives W[i,j] = k_b[i]·k[j] * decay
    // Solve X = I + W @ X  by back-substitution:
    //   X[:, 0, :] = e_0
    //   X[:, i, :] = e_i + sum_{j<i} W[i,j] * X[j, :]
    // Since CS=64 this is a 64-step loop on tensors of shape [BH, CS] each step
    let cs = kb.dim(1)?;
    let mut cols: Vec<Tensor> = Vec::with_capacity(cs);
    for i in 0..cs {
        let mut col = Tensor::zeros(...);   // [BH, CS] unit impulse at position i
        for j in 0..i {
            let w_ij = kb.i((.., i, j))?;  // [BH]
            col = (col + w_ij.unsqueeze(-1) * &cols[j])?;
        }
        cols.push(col);
    }
    Tensor::stack(&cols, 1)  // [BH, CS, CS]
}
```

**Call site in `apply_recurrence()`**:

```rust
fn apply_recurrence(&self, q, k, v, g, beta, ..., cache) -> Result<Tensor> {
    // CUDA path (existing, for decode seq_len==1 it's already fast)
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() {
        if seq_len == 1 {
            return gated_delta_rule_recurrence_cuda(...);  // existing fast path
        } else {
            return gated_delta_rule_chunked(q, k, v, g, beta, &mut cache.recurrent_state);
        }
    }
    // Metal path
    #[cfg(feature = "metal")]
    if q.device().is_metal() {
        if seq_len == 1 {
            return gated_delta_rule_recurrence_metal_typed(...);  // existing fast path
        } else {
            return gated_delta_rule_chunked(q, k, v, g, beta, &mut cache.recurrent_state);
        }
    }
    // CPU: always use chunked (replaces the current terrible O(seq_len) Rust loop)
    gated_delta_rule_chunked(q, k, v, g, beta, &mut cache.recurrent_state)
}
```

### Expected Gain

| Scenario | Current | After Phase 1 |
|----------|---------|---------------|
| CUDA prefill, T=2048 | 2048 serial GPU steps per GDN layer | 32 serial steps + GEMM |
| Metal prefill, T=2048 | 2048 serial GPU steps per GDN layer | 32 serial steps + GEMM |
| CPU prefill | O(T×Dk²) with per-token alloc | O(T/64 × 64² × Dv) BLAS |

Prefill throughput: **10–40× improvement** depending on sequence length and device.

---

## Phase 2 — Fused CUDA Chunked Kernel

### What & Why

Phase 1 uses Candle tensor ops for the chunked algorithm. While dramatically better than
token-by-token, Candle still dispatches ~20 separate CUDA kernels per chunk (matmuls,
elementwise ops). A single fused CUDA kernel can fuse the intra-chunk GEMM + decay +
state update into one kernel dispatch.

This mirrors llama.cpp's `build_delta_net_chunking` being backed by heavily-optimized
CUDA GEMM. This phase is a further ~2–4× speedup on top of Phase 1.

### Files to Change

- `mistralrs-core/src/cuda/gdn.cu` — add `gated_delta_rule_chunked_kernel`
- `mistralrs-core/src/cuda/ffi.rs` — declare `gated_delta_rule_chunked` extern
- `mistralrs-core/src/cuda/gdn.rs` — add `gated_delta_rule_chunked_cuda()` wrapper
- `mistralrs-core/src/models/deltanet.rs` — call new CUDA path in `apply_recurrence()`

### Kernel Design

```cuda
// One threadblock per (batch*head, chunk).
// Shared memory holds a [CS, Dk] tile of q/k and a [CS, Dv] tile of v.
// Registers hold the [Dk, Dv] state matrix for this head.
//
// Inner loop over CS=64 (compile-time constant) — fully unrolled with #pragma unroll.
// State update via register-resident outer product accumulation.

#define CS 64

__global__ void gated_delta_rule_chunked_kernel(
    const float * __restrict__ q,       // [BH, T, Dk]
    const float * __restrict__ k,       // [BH, T, Dk]
    const float * __restrict__ v,       // [BH, T, Dv]
    const float * __restrict__ g,       // [BH, T]
    const float * __restrict__ beta,    // [BH, T]
          float * __restrict__ output,  // [BH, T, Dv]
          float * __restrict__ states,  // [BH, Dk, Dv]  in/out
    int T, int Dk, int Dv
) {
    // blockIdx.x = bh, blockIdx.y = chunk
    const int bh    = blockIdx.x;
    const int chunk = blockIdx.y;
    const int dv    = threadIdx.x;   // each thread owns one Dv element

    // Load state into registers (Dk floats per thread)
    float s[MAX_DK];   // state[:, dv] for this Dv lane
    for (int dk = 0; dk < Dk; dk++)
        s[dk] = states[(bh * Dk + dk) * Dv + dv];

    __shared__ float sh_k[CS][MAX_DK];
    __shared__ float sh_q[CS][MAX_DK];
    __shared__ float sh_v[CS];    // one Dv lane loaded per sync

    float g_cs[CS];   // intra-chunk cumsum, per-thread in registers
    float acc = 0.f;
    // ... cumsum + decay mask computation ...

    #pragma unroll
    for (int i = 0; i < CS; i++) {
        float decay = expf(g_cs[CS-1] - g_cs[i]);
        float ki    = sh_k[i][...];
        float vi    = sh_v[i];     // for this dv lane
        float bi    = sh_beta[i];

        // State update: s += decay * ki * (vi - s·ki) * bi
        float sk = 0.f;
        #pragma unroll
        for (int dk = 0; dk < Dk; dk++) sk += s[dk] * sh_k[i][dk];
        float delta = (vi - sk) * bi;
        #pragma unroll
        for (int dk = 0; dk < Dk; dk++) s[dk] += decay * sh_k[i][dk] * delta;

        // Output: o[i, dv] = q[i] · s
        float o = 0.f;
        #pragma unroll
        for (int dk = 0; dk < Dk; dk++) o += sh_q[i][dk] * s[dk];
        output[(bh * T + chunk * CS + i) * Dv + dv] = o;
    }

    // Write back state
    for (int dk = 0; dk < Dk; dk++)
        states[(bh * Dk + dk) * Dv + dv] = s[dk];
}
```

Note: This kernel loops over CS=64 (compile-time) in a single GPU thread per (bh, dv)
position. This is naturally the "intra-chunk sequential scan" that must be sequential. The
inter-chunk parallelism comes from the grid dimension: all chunks can be launched together.

Actually, inter-chunk state updates are dependent (chunk N depends on chunk N-1's state),
so the chunks cannot be fully parallel. The correct approach is to launch one block per
(bh, chunk) but execute chunks in order. Alternatively, use the **prefix-sum trick**
(parallel scan) for full parallelism — but that requires storing intermediate states and
is more complex. For an initial implementation, sequential chunk dispatch is still
`seq_len/64 = ~32` kernel launches vs `seq_len = 2048`, a 64× reduction in serial work.

**Better approach**: fuse all chunks of one (bh) into a single block:

```cuda
// One block per bh. Block iterates over all n_chunks sequentially.
// Within each chunk, warps collaborate to compute the CS×CS inner products in parallel.
// State is register-resident across the full sequence for this block — never written to HBM
// until the end. This eliminates inter-chunk HBM traffic for the state.

__global__ void gdn_chunked_scan_kernel(...) {
    const int bh = blockIdx.x;
    // State lives in shared memory: [Dk, Dv] matrix
    __shared__ float state[MAX_DK][MAX_DV];
    // ... initialize from states[bh] ...

    for (int chunk = 0; chunk < n_chunks; chunk++) {
        // Load chunk slice into shared mem
        // Compute intra-chunk decay, kb, kq via warp-level matmul
        // Update output and state
    }
    // Write back state once
}
```

This eliminates `n_chunks` HBM round-trips for the state (currently the state is read/
written every token in the existing kernel).

### Expected Additional Gain (on top of Phase 1)

**2–4×** on CUDA prefill over Phase 1 Candle-op implementation, by eliminating kernel-
launch overhead and HBM state traffic between chunks.

---

## Phase 3 — Fused Metal Chunked Kernel

Same design as Phase 2 but for Metal (Apple Silicon). The existing Metal recurrence kernel
(`gdn.metal`) already has the register-resident state pattern — extend it with the chunked
outer loop.

### Files to Change

- `mistralrs-core/src/metal/gdn.metal` — add `gdn_chunked_scan` kernel
- `mistralrs-core/src/metal/gdn.rs` — add `gated_delta_rule_chunked_metal()` wrapper
- `mistralrs-core/src/models/deltanet.rs` — call from Metal branch in `apply_recurrence()`

### Key Differences from CUDA Version

- Use `threadgroup` memory instead of `__shared__`
- Metal has unified memory so HBM state traffic concern is less severe, but the kernel
  launch overhead is still eliminated (hundreds fewer dispatches per forward pass)
- Can leverage `simd_sum` and `simd_shuffle` for fast warp-equivalent reductions
- Use compile-time `constexpr uint CS = 64` for full loop unrolling

### Expected Additional Gain (on top of Phase 1)

**1.5–3×** on Metal prefill over Phase 1.

---

## Phase 4 — Reduce `.contiguous()` / Transpose Overhead in `deltanet.rs`

### What & Why

The current `apply_recurrence()` CUDA path (for decode, `seq_len==1`) executes:

```rust
let q_bh = q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?
            .reshape(...)?.contiguous()?;
// × 5 tensors = 10 GPU copy kernels per GDN layer per token
```

For a 14-layer hybrid model with 7 GDN layers, that's **70 unnecessary GPU copy kernels
per generated token** just to feed the recurrence kernel.

The fix is to store/pass tensors in the layout the kernel expects, avoiding late transposes.

### Files to Change

- `mistralrs-core/src/models/deltanet.rs` — reshape in `forward()` before calling
  `apply_recurrence()`, targeting the `[B*H, seq, D]` layout the CUDA/Metal kernels want
- `GdnLayerCache` — consider storing `conv_state` in kernel-native layout to avoid
  transpose on every decode step
- For the Metal decode path (seq_len==1): the fast-path reshape is already there but
  still does `g.reshape(...)` separately — fuse `g` and `beta` into one contiguous tensor

### Concrete Changes

```rust
// In GatedDeltaNet::forward(), before calling apply_recurrence():
// Current: q shape is [B, seq, H, Dk] and apply_recurrence does transpose inside
// Better: transpose once here and pass [B, H, seq, Dk] directly
let q = q.permute((0, 2, 1, 3))?.contiguous()?;  // done once
let k = k.permute((0, 2, 1, 3))?.contiguous()?;
let v = v.permute((0, 2, 1, 3))?.contiguous()?;
let g = g.permute((0, 2, 1))?.contiguous()?;
let beta = beta.permute((0, 2, 1))?.contiguous()?;
// ... then apply_recurrence gets them pre-transposed, no internal copies needed

// In GdnLayerCache::new(): store conv_state as [B, conv_dim, kernel] (already correct)
// but ensure it stays contiguous across cache updates to avoid copy-on-access
```

Also in `compute_gating()`: the CUDA/Metal fast paths already avoid dtype-converts, but
the CPU fallback does `a.unsqueeze(0)?.unsqueeze(0)?` which broadcasts to a huge shape.
Replace with a single `a.reshape([B, 1, seq, H])` to avoid the broadcast allocation.

### Expected Gain

**10–25% faster decode** across all devices.

---

## Phase 5 — Fuse GDN Input Projections

### What & Why

For Qwen3.5 (not world-size > 1), there are 4 separate matmuls per GDN layer per token:

```rust
// SplitQkvZa variant:
let proj_qkv = MatMul.qmethod_matmul(x, &**in_proj_qkv)?;  // [B, seq, key*2+val]
let z_full   = MatMul.qmethod_matmul(x, &**in_proj_z)?;    // [B, seq, val]
let b        = MatMul.qmethod_matmul(x, &**in_proj_b)?;    // [B, seq, H]
let a        = MatMul.qmethod_matmul(x, &**in_proj_a)?;    // [B, seq, H]
```

For decode (`seq_len=1`), this is 4 GEMV calls on the same `x`. They can be replaced by a
single fused matmul against the vertically-stacked weight matrix, then a narrow/split.

### Files to Change

- `mistralrs-core/src/models/deltanet.rs` — add `GdnProjection::FusedAll` variant
- `deltanet.rs` `load_qwen3_5()` — load stacked weight `in_proj_all = cat([qkv, z, b, a], dim=0)`
  during model init (done once, at load time)
- `project_inputs()` — single matmul then split via narrow

```rust
// New variant:
pub enum GdnProjection {
    // ... existing ...
    /// Fused: single weight [hidden, key*2 + val + val + H + H]
    FusedAll {
        in_proj: Arc<dyn QuantMethod>,
        qkv_end: usize,
        z_end:   usize,
        b_end:   usize,
        // a is the remainder
    },
}

// In project_inputs():
GdnProjection::FusedAll { in_proj, qkv_end, z_end, b_end } => {
    let all = MatMul.qmethod_matmul(x, &**in_proj)?;  // single GEMM/GEMV
    let proj_qkv = all.narrow(D::Minus1, 0,       *qkv_end)?;
    let z_full   = all.narrow(D::Minus1, *qkv_end, *z_end - *qkv_end)?;
    let b        = all.narrow(D::Minus1, *z_end,   *b_end - *z_end)?;
    let a        = all.narrow(D::Minus1, *b_end,   all.dim(D::Minus1)? - *b_end)?;
    // ... same reshape logic as before
}
```

**ISQ note**: The fused weight is a single `QuantMethod` tensor — ISQ applies to it
uniformly. The narrow/split after the matmul is zero-copy (view into output buffer).

**TP (world_size > 1) note**: Keep the existing `SplitQkvZaMerged` variant for TP; only
use `FusedAll` when `world_size == 1`.

### Expected Gain

**5–15% faster decode** (4 GEMV kernel launches → 1, plus 3 fewer scheduler round-trips).

---

## Phase 6 — CPU: Replace Manual Loops with BLAS-backed Candle Ops

### What & Why

The CPU fallback in both `causal_conv1d_full` and `gated_delta_rule_recurrence` uses
manually-written Rust `for` loops that allocate one `Tensor` per token. For a 2048-token
prompt with 7 GDN layers, that's ~28 000 Tensor heap allocations, all serialized.

Phase 1 already fixes `gated_delta_rule_recurrence` on CPU by routing through
`gated_delta_rule_chunked`. The remaining issue is `causal_conv1d_full`.

### Files to Change

- `mistralrs-core/src/models/deltanet.rs` — `causal_conv1d_full()` CPU path

### Algorithm

Replace the per-position loop with a single 1-D convolution expressed as a strided
matmul (a standard im2col-free convolution trick):

```rust
fn causal_conv1d_full_cpu(x_t: &Tensor, weight: &Tensor, kernel_size: usize) -> Result<Tensor> {
    // x_t: [B, conv_dim, seq_len]   weight: [conv_dim, kernel_size]
    // Build Toeplitz view: for each output position i, input window is x_t[..., i-K+1..=i]
    // Expressed as a single matmul: [B, conv_dim, seq_len] x [kernel_size] -> [B, conv_dim, seq_len]

    let (b, c, t) = x_t.dims3()?;
    let k = kernel_size;

    // Pad left with k-1 zeros: [B, c, t + k - 1]
    let pad = Tensor::zeros((b, c, k - 1), x_t.dtype(), x_t.device())?;
    let padded = Tensor::cat(&[pad, x_t.clone()], 2)?;  // [B, c, t+k-1]

    // Build windows tensor via as_strided / unfold:
    // windows[b, c, i, :] = padded[b, c, i..i+k]   shape [B, c, t, k]
    // In Candle this is: padded.unfold(2, k, 1)
    let windows = padded.unfold(2, k, 1)?;  // [B, c, t, k]

    // weight: [c, k] -> [c, k, 1] for conv
    // output[b, c, i] = sum_j windows[b, c, i, j] * weight[c, j]
    let w = weight.unsqueeze(0)?.unsqueeze(0)?;   // [1, 1, c, k] broadcast
    let out = (windows * w.transpose(2, 3)?)?.sum(D::Minus1)?;   // [B, c, t]
    candle_nn::ops::silu(&out)
}
```

If `Tensor::unfold` is not available in the Candle version in use, implement it with a
single `as_strided` call or via `narrow + stack` over kernel positions (still O(K) tensor
ops, far fewer than O(T)).

### Expected Gain

**5–20× faster CPU prefill** for `causal_conv1d_full` (eliminates T tensor allocations;
uses BLAS GEMM instead of scalar Rust loops).

---

## Implementation Order & Effort Estimates

| Phase | Effort | Backend(s) | ROI |
|-------|--------|-----------|-----|
| 1 — Chunked Candle tensor ops | Medium (3–5 days) | All | Highest — fixes prefill everywhere |
| 4 — Reduce contiguous/transpose | Small (1 day) | All | Immediate decode improvement |
| 5 — Fuse GDN input projections | Small (1–2 days) | All | Decode improvement |
| 6 — CPU conv1d loop → matmul | Small (1 day) | CPU | CPU prefill improvement |
| 2 — CUDA chunked kernel | Large (5–7 days) | CUDA | Further 2–4× on top of Phase 1 |
| 3 — Metal chunked kernel | Large (5–7 days) | Metal | Further 1.5–3× on top of Phase 1 |

**Recommended start**: Phase 1 + Phase 4 together. They are purely in Rust/Candle, require
no CUDA/Metal kernel work, and address the two largest performance gaps.

---

## Correctness Validation

For each phase, validate against the existing output at `seq_len=1` (decode) and at
`seq_len=32` (short prefill, where chunked = 1 chunk = same as sequential):

```bash
# Quick numerical equivalence check (add a test in mistralrs-core):
cargo test -p mistralrs-core gdn_chunked_matches_sequential -- --nocapture

# Full model output equivalence (need HF token):
TESTS_HF_TOKEN=... cargo test -p mistralrs-core qwen3_5_output_stable
```

The chunked algorithm is mathematically equivalent to the sequential algorithm — not an
approximation — so numerical output (in F32 arithmetic) should match to within floating-
point reordering tolerance (~1e-5 for BF16 models).

The `lower_tri_solve` back-substitution is exact for the CS=64 inner loop.

---

## Summary

The single most impactful change is **Phase 1** (chunked prefill via Candle tensor ops).
It turns the GDN prefill from a `seq_len`-serial GPU computation into a `seq_len/64`-
serial one with all intra-chunk work expressed as batched GEMM — exactly mirroring what
makes llama.cpp fast. Phases 2/3 add fused kernels for additional gains. Phases 4/5/6
are lightweight cleanup that recover decode performance.
