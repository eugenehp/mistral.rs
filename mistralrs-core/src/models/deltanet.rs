#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Shared GatedDeltaNet (linear attention) layer used by Qwen3Next, Qwen3.5, and Qwen3.5-MoE.
//!
//! This module provides the core DeltaNet recurrence and causal conv1d primitives that are common
//! across all Qwen3 hybrid-attention models.

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::Linear;
use mistralrs_quant::{QuantMethod, QuantizedConfig, RowParallelLayer, ShardedVarBuilder};
use std::sync::Arc;

use crate::device_map::DeviceMapper;
use crate::layers::MatMul;

// ====================== DeltaNet Config Trait ======================

/// Trait to abstract DeltaNet-relevant config fields from model-specific Config structs.
pub trait DeltaNetConfig {
    fn hidden_size(&self) -> usize;
    fn rms_norm_eps(&self) -> f64;
    fn linear_num_key_heads(&self) -> usize;
    fn linear_num_value_heads(&self) -> usize;
    fn linear_key_head_dim(&self) -> usize;
    fn linear_value_head_dim(&self) -> usize;
    fn linear_conv_kernel_dim(&self) -> usize;
    fn quantization_config(&self) -> &Option<QuantizedConfig>;

    /// Total key dimension = num_key_heads * key_head_dim
    fn linear_key_dim(&self) -> usize {
        self.linear_num_key_heads() * self.linear_key_head_dim()
    }

    /// Total value dimension = num_value_heads * value_head_dim
    fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads() * self.linear_value_head_dim()
    }

    /// Conv dim for GDN = key_dim * 2 + value_dim (q, k, v before split)
    fn linear_conv_dim(&self) -> usize {
        self.linear_key_dim() * 2 + self.linear_value_dim()
    }
}

// ====================== RMSNorm Gated (for GDN output) ======================

/// RMSNorm with gating: `rms_norm(x) * weight * silu(gate)`
pub struct RmsNormGated {
    pub weight: Tensor,
    pub eps: f64,
}

impl RmsNormGated {
    pub fn new(
        size: usize,
        eps: f64,
        vb: ShardedVarBuilder,
        isq_target_device: Option<&Device>,
    ) -> Result<Self> {
        let mut weight = vb.get(size, "weight")?;
        if let Some(target_dev) = isq_target_device {
            weight = weight.to_device(target_dev)?;
        }
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        // On Metal: use a single fused kernel that combines the 12-step Candle chain
        //   (to_dtype ×3, silu, sqr, mean_keepdim, +eps, sqrt, div, mul ×2, to_dtype)
        // into one cooperative GPU pass.
        #[cfg(feature = "metal")]
        if x.device().is_metal() {
            return crate::metal::gdn::gdn_rms_norm_gated_metal(
                x,
                gate,
                &self.weight,
                self.eps,
            );
        }

        // CPU / CUDA fallback:
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let gate = candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let out = normed
            .broadcast_mul(&self.weight.to_dtype(DType::F32)?)?
            .broadcast_mul(&gate)?;
        out.to_dtype(dtype)
    }
}

// ====================== GDN layer cache ======================

#[derive(Debug)]
pub struct GdnLayerCache {
    /// Conv state: (batch, conv_dim, kernel_size)
    pub conv_state: Tensor,
    /// Recurrent state: (batch, num_v_heads, head_k_dim, head_v_dim)
    pub recurrent_state: Tensor,
    pub seqlen_offset: usize,
}

impl GdnLayerCache {
    pub fn new(
        cfg: &dyn DeltaNetConfig,
        dtype: DType,
        device: &Device,
        world_size: usize,
    ) -> Result<Self> {
        let conv_dim = cfg.linear_conv_dim() / world_size;
        let conv_state = Tensor::zeros((1, conv_dim, cfg.linear_conv_kernel_dim()), dtype, device)?;
        // Store the recurrent state in F32 regardless of the model dtype.
        // The recurrence kernel always runs in F32, so keeping the state in F32
        // avoids two dtype-conversion kernels (BF16→F32, F32→BF16) on every
        // decode step for every GDN layer (e.g. 28 round-trips for a 14-layer hybrid).
        let recurrent_state = Tensor::zeros(
            (
                1,
                (cfg.linear_num_value_heads() / world_size).max(1),
                cfg.linear_key_head_dim(),
                cfg.linear_value_head_dim(),
            ),
            DType::F32,
            device,
        )?;
        Ok(Self {
            conv_state,
            recurrent_state,
            seqlen_offset: 0,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.conv_state = self.conv_state.zeros_like()?;
        self.recurrent_state = self.recurrent_state.zeros_like()?;
        self.seqlen_offset = 0;
        Ok(())
    }
}

impl Clone for GdnLayerCache {
    fn clone(&self) -> Self {
        Self {
            conv_state: self.conv_state.clone(),
            recurrent_state: self.recurrent_state.clone(),
            seqlen_offset: self.seqlen_offset,
        }
    }
}

// ====================== GDN math functions ======================

pub fn l2_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    // On Metal: use a single fused kernel that combines sqr → sum_keepdim →
    // broadcast_add(eps) → sqrt → recip → broadcast_mul in one GPU pass.
    // This reduces 6–8 GPU kernel dispatches to 1, saving ~196 dispatches
    // per decode token (2 calls × 14 GDN layers × 7 saved kernels).
    #[cfg(feature = "metal")]
    if x.device().is_metal() {
        return crate::metal::gdn::gdn_l2_norm_metal(x, eps as f32);
    }

    // CPU / CUDA fallback:
    let inv_norm = x
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .broadcast_add(&Tensor::new(eps as f32, x.device())?.to_dtype(x.dtype())?)?
        .sqrt()?
        .recip()?;
    x.broadcast_mul(&inv_norm)
}

pub fn softplus(x: &Tensor) -> Result<Tensor> {
    (Tensor::ones_like(x)? + x.exp()?)?.log()
}

// ====================== Causal mask helpers ======================

/// Build a [n, n] lower-triangular mask (1.0 on and below diagonal, 0.0 above).
/// Used by the chunked recurrence for intra-chunk causal attention.
fn lower_tri_mask(n: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            data[i * n + j] = 1.0;
        }
    }
    Tensor::from_slice(&data, (n, n), device)
}

/// Build a [n, n] strictly-lower-triangular mask (1.0 below diagonal, 0.0 on/above).
fn strict_lower_tri_mask(n: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; n * n];
    for i in 0..n {
        for j in 0..i {
            data[i * n + j] = 1.0;
        }
    }
    Tensor::from_slice(&data, (n, n), device)
}

// ====================== Chunk-parallel gated delta rule ======================

/// Chunk-parallel gated delta rule recurrence (exact, no approximation).
///
/// Replaces the O(seq_len) token-by-token loop with O(n_chunks) sequential
/// state updates. All intra-chunk computation is expressed as batch GEMM,
/// making full use of BLAS on CPU and cuBLAS/MPS on GPU.
///
/// This mirrors llama.cpp's `build_delta_net_chunking` (CS = `chunk_size`),
/// including the exact unit-lower-triangular back-substitution.
///
/// # Tensor layout (all F32)
/// * `q`, `k` : `[bh, T, Dk]`  (`q` is already scaled by `1/√Dk`)
/// * `v`      : `[bh, T, Dv]`
/// * `g`      : `[bh, T]`  per-step log-decay
/// * `beta`   : `[bh, T]`  per-step correction scale
/// * `state`  : `[bh, Dk, Dv]`  updated in-place on every chunk boundary
///
/// Returns `output` of shape `[bh, T, Dv]`.
pub fn gated_delta_rule_chunked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
    chunk_size: usize,
) -> Result<Tensor> {
    let (_bh, t_total, _dk) = q.dims3()?;
    let _dv = v.dim(2)?;
    let n_chunks = t_total.div_ceil(chunk_size);
    let device = q.device().clone();
    let mut chunks_out: Vec<Tensor> = Vec::with_capacity(n_chunks);

    for ci in 0..n_chunks {
        let start = ci * chunk_size;
        let cs = (t_total - start).min(chunk_size);

        // ── Slice chunk [bh, cs, D] ─────────────────────────────────────────
        let q_c = q.narrow(1, start, cs)?;
        let k_c = k.narrow(1, start, cs)?;
        let v_c = v.narrow(1, start, cs)?;
        let g_c = g.narrow(1, start, cs)?;
        let beta_c = beta.narrow(1, start, cs)?;

        // ── 1. Cumulative log-decay [bh, cs] ────────────────────────────────
        // g_cs[b, i] = Σ_{t=0..=i} g_c[b, t]
        let g_cs = g_c.cumsum(1)?;

        // ── 2. Intra-chunk decay matrix [bh, cs, cs] ────────────────────────
        // decay[b, i, j] = exp(g_cs[b,i] - g_cs[b,j])  (j→i multiplicative decay)
        let g_i = g_cs.unsqueeze(2)?; // [bh, cs, 1]
        let g_j = g_cs.unsqueeze(1)?; // [bh, 1, cs]
        let raw_decay = (g_i.broadcast_sub(&g_j)?).exp()?; // [bh, cs, cs]

        let causal_mask = lower_tri_mask(cs, &device)?;        // incl. diagonal
        let strict_mask = strict_lower_tri_mask(cs, &device)?; // excl. diagonal
        let decay = raw_decay.broadcast_mul(&causal_mask)?;         // [bh, cs, cs]
        let decay_strict = raw_decay.broadcast_mul(&strict_mask)?;  // [bh, cs, cs]

        // ── 3. Beta-weighted tensors ─────────────────────────────────────────
        let k_b = (&k_c * beta_c.unsqueeze(2)?)?; // [bh, cs, dk]
        let v_b = (&v_c * beta_c.unsqueeze(2)?)?; // [bh, cs, dv]

        // ── 4. RHS of the triangular system ──────────────────────────────────
        // rhs[b, i] = v_b[b,i] − exp(g_cs[b,i]) · (state ← k_c[b,i])
        //
        // k_c @ state: [bh, cs, dk] @ [bh, dk, dv] = [bh, cs, dv]
        //   entry [b,i,v] = Σ_j k_c[b,i,j] * state[b,j,v]   (state retrieval)
        let sk = k_c.contiguous()?.matmul(state)?; // [bh, cs, dv]
        let g_exp = g_cs.exp()?.unsqueeze(2)?; // [bh, cs, 1]
        let rhs = (v_b - sk.broadcast_mul(&g_exp)?)?; // [bh, cs, dv]

        // ── 5. A matrix (strictly lower triangular) ──────────────────────────
        // A[b, i, j] = decay_strict[b,i,j] · (k_b[b,i] · k_c[b,j])
        //
        // kbk = k_b @ k_c^T  [bh, cs, cs],  entry [b,i,j] = k_b[b,i] · k_c[b,j]
        let kbk = k_b.contiguous()?.matmul(&k_c.t()?.contiguous()?)?; // [bh, cs, cs]
        let a_mat = (&kbk * &decay_strict)?; // [bh, cs, cs]

        // ── 6. Back-substitution: δ = (I + A)^{-1} · rhs ────────────────────
        // A is strictly lower triangular ⟹ (I+A) is unit lower triangular.
        // δ[i] = rhs[i] − Σ_{j<i} A[i,j] · δ[j]
        //
        // This is a cs-step sequential loop, but cs = CHUNK_SIZE ≪ seq_len.
        let mut delta_cols: Vec<Tensor> = Vec::with_capacity(cs);
        // i = 0: no correction (no elements to the left)
        delta_cols.push(rhs.narrow(1, 0, 1)?.squeeze(1)?); // [bh, dv]

        for i in 1..cs {
            let rhs_i = rhs.narrow(1, i, 1)?.squeeze(1)?; // [bh, dv]
            // a_row[b, j] = A[b, i, j] for j in 0..i  [bh, i]
            let a_row = a_mat
                .narrow(1, i, 1)?
                .squeeze(1)?
                .narrow(1, 0, i)?; // [bh, i]
            // d_prev: stack of δ[0..i]  [bh, i, dv]
            let d_prev = Tensor::stack(&delta_cols[..i], 1)?;
            // corr = a_row @ d_prev → [bh, 1, i] @ [bh, i, dv] = [bh, 1, dv] → [bh, dv]
            let corr = a_row.unsqueeze(1)?.matmul(&d_prev)?.squeeze(1)?;
            delta_cols.push((rhs_i - corr)?);
        }
        let delta = Tensor::stack(&delta_cols, 1)?; // [bh, cs, dv]

        // ── 7. Output ────────────────────────────────────────────────────────
        // o[b,i] = exp(g_cs[i]) · (state @ q_c[i])
        //        + Σ_{j≤i} decay[i,j] · (k_c[j] · q_c[i]) · δ[j]
        //
        // o_state = (q_c @ state) * g_exp        [bh, cs, dv]
        // kq      = q_c @ k_c^T                  [bh, cs, cs]  (q already scaled)
        // o_intra = (kq * decay) @ delta          [bh, cs, dv]
        let o_state = q_c
            .contiguous()?
            .matmul(state)?
            .broadcast_mul(&g_exp)?; // [bh, cs, dv]
        let kq = q_c
            .contiguous()?
            .matmul(&k_c.t()?.contiguous()?)?; // [bh, cs, cs]
        let o_intra = (&kq * &decay)?.matmul(&delta)?; // [bh, cs, dv]
        chunks_out.push((o_state + o_intra)?);

        // ── 8. State update ──────────────────────────────────────────────────
        // state_new = exp(g_cs[-1]) · state + K_decay^T @ δ
        //
        // K_decay[b, j] = k_c[b, j] * exp(g_cs[-1] - g_cs[j])
        //   decay_end[b, j] = exp(g_cs[b, cs-1] - g_cs[b, j])  [bh, cs]
        let g_last = g_cs.narrow(1, cs - 1, 1)?; // [bh, 1]
        let decay_end = (g_last.broadcast_sub(&g_cs)?).exp()?; // [bh, cs]
        let k_decay = (&k_c * decay_end.unsqueeze(2)?)?; // [bh, cs, dk]
        let state_delta = k_decay.t()?.contiguous()?.matmul(&delta)?; // [bh, dk, dv]
        *state =
            (state.broadcast_mul(&g_last.exp()?.unsqueeze(2)?)? + state_delta)?;
    }

    Tensor::cat(&chunks_out, 1) // [bh, T, Dv]
}

/// Recurrent gated delta rule — token-by-token sequential fallback.
///
/// Kept as a correctness reference and for very short sequences (T ≤ 4) where
/// the chunked path's GEMM setup overhead exceeds its benefit.
///
/// q, k: (batch, seq, num_v_heads, head_k_dim)
/// v:    (batch, seq, num_v_heads, head_v_dim)
/// g:    (batch, seq, num_v_heads)
/// beta: (batch, seq, num_v_heads)
/// state: (batch, num_v_heads, head_k_dim, head_v_dim)
///
/// Returns: (batch, seq, num_v_heads, head_v_dim)
#[allow(dead_code)]
pub fn gated_delta_rule_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let dtype = q.dtype();
    let k_head_dim = q.dim(D::Minus1)?;
    let scale = 1.0 / (k_head_dim as f64).sqrt();

    let q = (q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)? * scale)?;
    let k = k.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let v = v.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let g = g.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let beta = beta.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;

    let seq_len = q.dim(2)?;
    let mut s = state.to_dtype(DType::F32)?;
    let mut outputs = Vec::with_capacity(seq_len);

    for i in 0..seq_len {
        let q_t = q.i((.., .., i, ..))?;
        let k_t = k.i((.., .., i, ..))?;
        let v_t = v.i((.., .., i, ..))?;
        let g_t = g.i((.., .., i))?;
        let beta_t = beta.i((.., .., i))?;

        let decay = g_t.exp()?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
        s = s.broadcast_mul(&decay)?;

        let k_exp = k_t.unsqueeze(D::Minus1)?;
        let kv_mem = s.broadcast_mul(&k_exp)?.sum(2)?;

        let beta_exp = beta_t.unsqueeze(D::Minus1)?;
        let delta = (v_t - kv_mem)?.broadcast_mul(&beta_exp)?;

        let outer = k_exp.broadcast_mul(&delta.unsqueeze(2)?)?;
        s = (s + outer)?;

        let q_exp = q_t.unsqueeze(D::Minus1)?;
        let y_t = s.broadcast_mul(&q_exp)?.sum(2)?;

        outputs.push(y_t);
    }

    *state = s.to_dtype(state.dtype())?;

    let out = Tensor::stack(&outputs, 2)?;
    out.transpose(1, 2)?.contiguous()?.to_dtype(dtype)
}

// ====================== GDN Projection variants ======================

/// Projection strategy for GDN input. Qwen3Next and Qwen3.5 differ in how they pack weights.
#[allow(dead_code)]
pub enum GdnProjection {
    /// Qwen3Next: fused in_proj_qkvz (key_dim*2 + value_dim*2) + in_proj_ba (num_v_heads*2)
    FusedQkvzBa {
        in_proj_qkvz: Arc<dyn QuantMethod>,
        in_proj_ba: Arc<dyn QuantMethod>,
    },
    /// Qwen3.5 world_size==1: all four projections merged into one weight for a
    /// single matmul dispatch.  Output split offsets are stored alongside.
    ///
    /// Layout of the fused output dim:
    ///   [0 .. qkv_end)    → q, k, v  (key_dim*2 + value_dim)
    ///   [qkv_end .. z_end) → z       (value_dim)
    ///   [z_end .. b_end)   → b       (num_v_heads)
    ///   [b_end .. end)     → a       (num_v_heads)
    FusedAll {
        in_proj: Arc<dyn QuantMethod>,
        qkv_end: usize,
        z_end: usize,
        b_end: usize,
    },
    /// Qwen3.5: split in_proj_qkv (key_dim*2 + value_dim) + in_proj_z (value_dim) + in_proj_b (num_v_heads) + in_proj_a (num_v_heads)
    SplitQkvZa {
        in_proj_qkv: Arc<dyn QuantMethod>,
        in_proj_z: Arc<dyn QuantMethod>,
        in_proj_b: Arc<dyn QuantMethod>,
        in_proj_a: Arc<dyn QuantMethod>,
    },
    /// Qwen3.5 TP-safe split for packed in_proj_qkv [q|k|v].
    SplitQkvZaMerged {
        in_proj_q: Arc<dyn QuantMethod>,
        in_proj_k: Arc<dyn QuantMethod>,
        in_proj_v: Arc<dyn QuantMethod>,
        in_proj_z: Arc<dyn QuantMethod>,
        in_proj_b: Arc<dyn QuantMethod>,
        in_proj_a: Arc<dyn QuantMethod>,
    },
}

impl GdnProjection {
    /// Return mutable references to every ISQ-quantisable projection weight.
    pub fn isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        match self {
            GdnProjection::FusedQkvzBa {
                in_proj_qkvz,
                in_proj_ba,
            } => vec![in_proj_qkvz, in_proj_ba],
            GdnProjection::FusedAll { in_proj, .. } => vec![in_proj],
            GdnProjection::SplitQkvZa {
                in_proj_qkv,
                in_proj_z,
                in_proj_b,
                in_proj_a,
            } => vec![in_proj_qkv, in_proj_z, in_proj_b, in_proj_a],
            GdnProjection::SplitQkvZaMerged {
                in_proj_q,
                in_proj_k,
                in_proj_v,
                in_proj_z,
                in_proj_b,
                in_proj_a,
            } => vec![in_proj_q, in_proj_k, in_proj_v, in_proj_z, in_proj_b, in_proj_a],
        }
    }
}

// ====================== Gated Delta Net layer ======================

/// Projected outputs from the GDN input projections.
/// z is 4D (batch, seq, num_v_heads, head_v_dim), others are flat for conv.
struct GdnProjected {
    /// (batch, seq, key_dim)
    q: Tensor,
    /// (batch, seq, key_dim)
    k: Tensor,
    /// (batch, seq, value_dim)
    v_flat: Tensor,
    /// (batch, seq, num_v_heads, head_v_dim) — gating signal for norm
    z: Tensor,
    /// (batch, seq, num_v_heads)
    b: Tensor,
    /// (batch, seq, num_v_heads)
    a: Tensor,
    /// Pre-concatenated [q|k|v_flat] slice produced by FusedAll.
    ///
    /// When present (FusedAll path only), this is the first `qkv_end` columns
    /// of the fused matmul output — already contiguous — so `forward()` can
    /// pass it straight to causal_conv1d without an extra `Tensor::cat` copy.
    qkv_for_conv: Option<Tensor>,
    /// Full fused projection output [B, S, total_out_dim] (FusedAll path only).
    ///
    /// Carried through so `forward()` can pass it directly to the strided
    /// gating kernel, which reads b and a via column offsets instead of
    /// requiring two `contiguous()` copy dispatches on non-contiguous narrow views.
    proj_all: Option<Tensor>,
}

pub struct GatedDeltaNet {
    pub projection: GdnProjection,
    pub conv1d_weight: Tensor,
    pub dt_bias: Tensor,
    pub a_log: Tensor,
    /// Cached F32 copy of a_log — avoids a to_dtype dispatch every forward step.
    pub a_log_f32: Tensor,
    /// Cached F32 copy of dt_bias — avoids a to_dtype dispatch every forward step.
    pub dt_bias_f32: Tensor,
    pub norm: RmsNormGated,
    pub out_proj: Arc<dyn QuantMethod>,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_kernel_size: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

impl GatedDeltaNet {
    /// Return mutable references to every quantisable weight in this GDN layer,
    /// including all input projections.  Call this from the model's `get_layers()`
    /// so that ISQ applies to input projections too (4× bandwidth reduction).
    pub fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        let mut v = self.projection.isq_layers();
        v.push(&mut self.out_proj);
        v
    }

    /// Load GDN layer with fused Qwen3Next projection (in_proj_qkvz + in_proj_ba).
    #[allow(dead_code)]
    pub fn load_qwen3next(
        vb: ShardedVarBuilder,
        cfg: &dyn DeltaNetConfig,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let isq_target_device = if loading_isq {
            mapper.device_for(layer_idx, false).cloned()
        } else {
            None
        };

        let world_size = comm.world_size();
        let num_k_heads_global = cfg.linear_num_key_heads();
        let num_v_heads_global = cfg.linear_num_value_heads();

        if !num_v_heads_global.is_multiple_of(world_size)
            || !num_k_heads_global.is_multiple_of(world_size)
        {
            candle_core::bail!(
                "linear attention heads must be divisible by tensor parallel world_size (num_v_heads={}, num_k_heads={}, world_size={})",
                num_v_heads_global,
                num_k_heads_global,
                world_size
            );
        }

        let num_k_heads = num_k_heads_global / world_size;
        let num_v_heads = num_v_heads_global / world_size;

        let head_k_dim = cfg.linear_key_head_dim();
        let head_v_dim = cfg.linear_value_head_dim();
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;

        let key_dim_global = num_k_heads_global * head_k_dim;
        let value_dim_global = num_v_heads_global * head_v_dim;

        let conv_kernel_size = cfg.linear_conv_kernel_dim();

        let vb_la = mapper.set_device(layer_idx, vb.pp("linear_attn"), loading_isq);

        let qkvz_out_global = key_dim_global * 2 + value_dim_global * 2;
        let in_proj_qkvz = mistralrs_quant::ColumnParallelLayer::new(
            cfg.hidden_size(),
            qkvz_out_global,
            cfg.quantization_config(),
            false,
            comm,
            vb_la.pp("in_proj_qkvz"),
        )?;
        let in_proj_ba = mistralrs_quant::ColumnParallelLayer::new(
            cfg.hidden_size(),
            num_v_heads_global * 2,
            cfg.quantization_config(),
            false,
            comm,
            vb_la.pp("in_proj_ba"),
        )?;

        let conv_dim_global = key_dim_global * 2 + value_dim_global;
        let mut conv1d_weight =
            vb_la.get((conv_dim_global, 1, conv_kernel_size), "conv1d.weight")?;

        let sd = mistralrs_quant::Shard::Simple {
            dim: 0,
            rank: comm.rank(),
            world_size: comm.world_size(),
        };
        let mut dt_bias = vb_la.get_with_hints((num_v_heads_global,), "dt_bias", sd)?;
        let mut a_log = vb_la.get_with_hints((num_v_heads_global,), "A_log", sd)?;

        let rank = comm.rank();
        let q_start = rank * key_dim;
        let k_start = key_dim_global + rank * key_dim;
        let v_start = key_dim_global * 2 + rank * value_dim;
        let q_w = conv1d_weight.narrow(0, q_start, key_dim)?;
        let k_w = conv1d_weight.narrow(0, k_start, key_dim)?;
        let v_w = conv1d_weight.narrow(0, v_start, value_dim)?;
        conv1d_weight = candle_core::Tensor::cat(&[&q_w, &k_w, &v_w], 0)?;

        if let Some(ref target_dev) = isq_target_device {
            conv1d_weight = conv1d_weight.to_device(target_dev)?;
            dt_bias = dt_bias.to_device(target_dev)?;
            a_log = a_log.to_device(target_dev)?;
        }

        let norm = RmsNormGated::new(
            head_v_dim,
            cfg.rms_norm_eps(),
            vb_la.pp("norm"),
            isq_target_device.as_ref(),
        )?;

        let out_proj = RowParallelLayer::new(
            value_dim,
            cfg.hidden_size(),
            cfg.quantization_config(),
            false,
            comm,
            vb_la.pp("out_proj"),
        )?;

        let a_log_f32 = a_log.to_dtype(DType::F32)?;
        let dt_bias_f32 = dt_bias.to_dtype(DType::F32)?;

        Ok(Self {
            projection: GdnProjection::FusedQkvzBa {
                in_proj_qkvz,
                in_proj_ba,
            },
            conv1d_weight,
            dt_bias,
            a_log,
            a_log_f32,
            dt_bias_f32,
            norm,
            out_proj,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel_size,
            key_dim,
            value_dim,
        })
    }

    /// Load GDN layer with split Qwen3.5 projection (in_proj_qkv + in_proj_z + in_proj_b + in_proj_a).
    pub fn load_qwen3_5(
        vb: ShardedVarBuilder,
        cfg: &dyn DeltaNetConfig,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let isq_target_device = if loading_isq {
            mapper.device_for(layer_idx, false).cloned()
        } else {
            None
        };

        let world_size = comm.world_size();
        let num_k_heads_global = cfg.linear_num_key_heads();
        let num_v_heads_global = cfg.linear_num_value_heads();

        if !num_v_heads_global.is_multiple_of(world_size)
            || !num_k_heads_global.is_multiple_of(world_size)
        {
            candle_core::bail!(
                "linear attention heads must be divisible by tensor parallel world_size (num_v_heads={}, num_k_heads={}, world_size={})",
                num_v_heads_global,
                num_k_heads_global,
                world_size
            );
        }

        let num_k_heads = num_k_heads_global / world_size;
        let num_v_heads = num_v_heads_global / world_size;

        let head_k_dim = cfg.linear_key_head_dim();
        let head_v_dim = cfg.linear_value_head_dim();
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;

        let key_dim_global = num_k_heads_global * head_k_dim;
        let value_dim_global = num_v_heads_global * head_v_dim;

        let conv_kernel_size = cfg.linear_conv_kernel_dim();

        let vb_la = mapper.set_device(layer_idx, vb.pp("linear_attn"), loading_isq);

        let qkv_out_global = key_dim_global * 2 + value_dim_global;
        let in_proj_z = mistralrs_quant::ColumnParallelLayer::new(
            cfg.hidden_size(),
            value_dim_global,
            cfg.quantization_config(),
            false,
            comm,
            vb_la.pp("in_proj_z"),
        )?;
        let in_proj_b = mistralrs_quant::ColumnParallelLayer::new(
            cfg.hidden_size(),
            num_v_heads_global,
            cfg.quantization_config(),
            false,
            comm,
            vb_la.pp("in_proj_b"),
        )?;
        let in_proj_a = mistralrs_quant::ColumnParallelLayer::new(
            cfg.hidden_size(),
            num_v_heads_global,
            cfg.quantization_config(),
            false,
            comm,
            vb_la.pp("in_proj_a"),
        )?;

        let conv_dim_global = key_dim_global * 2 + value_dim_global;
        let mut conv1d_weight =
            vb_la.get((conv_dim_global, 1, conv_kernel_size), "conv1d.weight")?;

        // Learned GDN parameters ()
        let sd = mistralrs_quant::Shard::Simple {
            dim: 0,
            rank: comm.rank(),
            world_size: comm.world_size(),
        };
        let mut dt_bias = vb_la.get_with_hints((num_v_heads_global,), "dt_bias", sd)?;
        let mut a_log = vb_la.get_with_hints((num_v_heads_global,), "A_log", sd)?;

        let rank = comm.rank();
        let q_start = rank * key_dim;
        let k_start = key_dim_global + rank * key_dim;
        let v_start = key_dim_global * 2 + rank * value_dim;
        let q_w = conv1d_weight.narrow(0, q_start, key_dim)?;
        let k_w = conv1d_weight.narrow(0, k_start, key_dim)?;
        let v_w = conv1d_weight.narrow(0, v_start, value_dim)?;
        conv1d_weight = candle_core::Tensor::cat(&[&q_w, &k_w, &v_w], 0)?;

        if let Some(ref target_dev) = isq_target_device {
            conv1d_weight = conv1d_weight.to_device(target_dev)?;
            dt_bias = dt_bias.to_device(target_dev)?;
            a_log = a_log.to_device(target_dev)?;
        }

        let norm = RmsNormGated::new(
            head_v_dim,
            cfg.rms_norm_eps(),
            vb_la.pp("norm"),
            isq_target_device.as_ref(),
        )?;

        let out_proj = RowParallelLayer::new(
            value_dim,
            cfg.hidden_size(),
            cfg.quantization_config(),
            false,
            comm,
            vb_la.pp("out_proj"),
        )?;

        let projection = if world_size > 1 {
            // Tensor-parallel: keep QKV split so each rank owns its shard.
            let merged_qkv = mistralrs_quant::ColumnParallelLayer::new_merged_chunks(
                cfg.hidden_size(),
                qkv_out_global,
                vec![key_dim_global, key_dim_global, value_dim_global],
                cfg.quantization_config(),
                comm,
                vb_la.pp("in_proj_qkv"),
            )?;
            if merged_qkv.len() != 3 {
                candle_core::bail!(
                    "Expected 3 merged chunks for in_proj_qkv, got {}",
                    merged_qkv.len()
                );
            }
            GdnProjection::SplitQkvZaMerged {
                in_proj_q: merged_qkv[0].clone(),
                in_proj_k: merged_qkv[1].clone(),
                in_proj_v: merged_qkv[2].clone(),
                in_proj_z,
                in_proj_b,
                in_proj_a,
            }
        } else {
            let in_proj_qkv = mistralrs_quant::ColumnParallelLayer::new(
                cfg.hidden_size(),
                qkv_out_global,
                cfg.quantization_config(),
                false,
                comm,
                vb_la.pp("in_proj_qkv"),
            )?;

            // ── Projection fusion (world_size == 1, unquantized weights) ────
            //
            // Fuse qkv + z + b + a into one contiguous weight matrix so that
            // project_inputs() issues a single matmul instead of four.  This
            // cuts ~3 Metal kernel dispatches per GDN layer per decode step.
            //
            // We only attempt fusion when all four weights are unquantized
            // (i.e. plain BF16/F16/F32 tensors).  Quantized models (ISQ,
            // GGUF, AWQ, …) fall back to the split layout unchanged.
            let try_fuse = || -> Result<GdnProjection> {
                let (w_qkv, _) = in_proj_qkv
                    .unquant_weight_bias()
                    .ok_or_else(|| candle_core::Error::Msg("not unquantized".into()))?;
                let (w_z, _) = in_proj_z
                    .unquant_weight_bias()
                    .ok_or_else(|| candle_core::Error::Msg("not unquantized".into()))?;
                let (w_b, _) = in_proj_b
                    .unquant_weight_bias()
                    .ok_or_else(|| candle_core::Error::Msg("not unquantized".into()))?;
                let (w_a, _) = in_proj_a
                    .unquant_weight_bias()
                    .ok_or_else(|| candle_core::Error::Msg("not unquantized".into()))?;

                // Stack rows: [qkv_out | value_dim | num_v_heads | num_v_heads] × hidden
                let fused = candle_core::Tensor::cat(&[&w_qkv, &w_z, &w_b, &w_a], 0)?;
                let qkv_end = w_qkv.dim(0)?;
                let z_end = qkv_end + w_z.dim(0)?;
                let b_end = z_end + w_b.dim(0)?;

                let in_proj = mistralrs_quant::ReplicatedLayer::from_linear(
                    Linear::new(fused, None),
                )?;
                Ok(GdnProjection::FusedAll {
                    in_proj,
                    qkv_end,
                    z_end,
                    b_end,
                })
            };

            match try_fuse() {
                Ok(proj) => proj,
                Err(_) => GdnProjection::SplitQkvZa {
                    in_proj_qkv,
                    in_proj_z,
                    in_proj_b,
                    in_proj_a,
                },
            }
        };

        let a_log_f32 = a_log.to_dtype(DType::F32)?;
        let dt_bias_f32 = dt_bias.to_dtype(DType::F32)?;

        Ok(Self {
            projection,
            conv1d_weight,
            dt_bias,
            a_log,
            a_log_f32,
            dt_bias_f32,
            norm,
            out_proj,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel_size,
            key_dim,
            value_dim,
        })
    }

    /// Project inputs and unpack into (q, k, v_flat, z, b, a) based on projection variant.
    fn project_inputs(&self, x: &Tensor) -> Result<GdnProjected> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let v_per_group = self.num_v_heads / self.num_k_heads;

        match &self.projection {
            GdnProjection::FusedAll {
                in_proj,
                qkv_end,
                z_end,
                b_end,
            } => {
                // Single matmul for all four projections.
                let all = MatMul.qmethod_matmul(x, &**in_proj)?; // [B, S, total_out]
                let proj_qkv = all.narrow(D::Minus1, 0, *qkv_end)?;
                let z_full = all.narrow(D::Minus1, *qkv_end, *z_end - *qkv_end)?;
                let b = all.narrow(D::Minus1, *z_end, *b_end - *z_end)?;
                let a_total = all.dim(D::Minus1)? - *b_end;
                let a = all.narrow(D::Minus1, *b_end, a_total)?;

                let q = proj_qkv.narrow(D::Minus1, 0, self.key_dim)?;
                let k = proj_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
                let v_flat = proj_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;
                let z = z_full.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

                Ok(GdnProjected {
                    q,
                    k,
                    v_flat,
                    z,
                    b,
                    a,
                    // proj_qkv is already the contiguous [q|k|v_flat] slice —
                    // pass it through so forward() can skip the cat() copy.
                    qkv_for_conv: Some(proj_qkv),
                    // Full projection output: used by the strided gating kernel
                    // to read b and a without a contiguous-copy dispatch.
                    proj_all: Some(all),
                })
            }
            GdnProjection::FusedQkvzBa {
                in_proj_qkvz,
                in_proj_ba,
            } => {
                let mixed_qkvz = MatMul.qmethod_matmul(x, &**in_proj_qkvz)?;
                let mixed_ba = MatMul.qmethod_matmul(x, &**in_proj_ba)?;

                let group_size_qkvz = 2 * self.head_k_dim + 2 * v_per_group * self.head_v_dim;
                let mixed_qkvz =
                    mixed_qkvz.reshape((batch_size, seq_len, self.num_k_heads, group_size_qkvz))?;

                let group_size_ba = 2 * v_per_group;
                let mixed_ba =
                    mixed_ba.reshape((batch_size, seq_len, self.num_k_heads, group_size_ba))?;

                let mut offset = 0;
                let q = mixed_qkvz.narrow(D::Minus1, offset, self.head_k_dim)?;
                offset += self.head_k_dim;
                let k = mixed_qkvz.narrow(D::Minus1, offset, self.head_k_dim)?;
                offset += self.head_k_dim;
                let v = mixed_qkvz.narrow(D::Minus1, offset, v_per_group * self.head_v_dim)?;
                offset += v_per_group * self.head_v_dim;
                let z = mixed_qkvz.narrow(D::Minus1, offset, v_per_group * self.head_v_dim)?;

                let b = mixed_ba.narrow(D::Minus1, 0, v_per_group)?;
                let a = mixed_ba.narrow(D::Minus1, v_per_group, v_per_group)?;

                // Reshape to per-head
                let v = v.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;
                let z = z.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;
                let b = b.reshape((batch_size, seq_len, self.num_v_heads))?;
                let a = a.reshape((batch_size, seq_len, self.num_v_heads))?;

                let q = q.reshape((batch_size, seq_len, self.key_dim))?;
                let k = k.reshape((batch_size, seq_len, self.key_dim))?;
                let v_flat = v.reshape((batch_size, seq_len, self.value_dim))?;

                Ok(GdnProjected {
                    q,
                    k,
                    v_flat,
                    z,
                    b,
                    a,
                    qkv_for_conv: None,
                    proj_all: None,
                })
            }
            GdnProjection::SplitQkvZa {
                in_proj_qkv,
                in_proj_z,
                in_proj_b,
                in_proj_a,
            } => {
                let proj_qkv = MatMul.qmethod_matmul(x, &**in_proj_qkv)?;
                let z_full = MatMul.qmethod_matmul(x, &**in_proj_z)?;
                let b = MatMul.qmethod_matmul(x, &**in_proj_b)?;
                let a = MatMul.qmethod_matmul(x, &**in_proj_a)?;

                let q = proj_qkv.narrow(D::Minus1, 0, self.key_dim)?;
                let k = proj_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
                let v_flat = proj_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

                let z = z_full.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

                Ok(GdnProjected {
                    q,
                    k,
                    v_flat,
                    z,
                    b,
                    a,
                    qkv_for_conv: None,
                    proj_all: None,
                })
            }
            GdnProjection::SplitQkvZaMerged {
                in_proj_q,
                in_proj_k,
                in_proj_v,
                in_proj_z,
                in_proj_b,
                in_proj_a,
            } => {
                let q = MatMul.qmethod_matmul(x, &**in_proj_q)?;
                let k = MatMul.qmethod_matmul(x, &**in_proj_k)?;
                let v_flat = MatMul.qmethod_matmul(x, &**in_proj_v)?;
                let z_full = MatMul.qmethod_matmul(x, &**in_proj_z)?;
                let b = MatMul.qmethod_matmul(x, &**in_proj_b)?;
                let a = MatMul.qmethod_matmul(x, &**in_proj_a)?;

                let z = z_full.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

                Ok(GdnProjected {
                    q,
                    k,
                    v_flat,
                    z,
                    b,
                    a,
                    qkv_for_conv: None,
                    proj_all: None,
                })
            }
        }
    }

    /// Run the full GDN forward pass.
    pub fn forward(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (batch_size, seq_len, _hidden) = x.dims3()?;
        let dtype = x.dtype();
        let v_per_group = self.num_v_heads / self.num_k_heads;

        // 1. Project input
        let projected = self.project_inputs(x)?;
        let GdnProjected {
            q,
            k,
            v_flat,
            z,
            b,
            a,
            qkv_for_conv,
            proj_all,
        } = projected;

        // 2. Concatenate q, k, v for conv1d: (batch, seq, conv_dim)
        //
        // FusedAll already gives us a contiguous [q|k|v_flat] slice — reuse it
        // directly and skip the Metal copy kernel that cat() would issue.
        let mixed_qkv = match qkv_for_conv {
            Some(raw) => raw,
            None => Tensor::cat(&[&q, &k, &v_flat], D::Minus1)?,
        };

        // 3. Apply causal conv1d (includes silu activation)
        let mixed_qkv = if cache.seqlen_offset > 0 && seq_len == 1 {
            self.causal_conv1d_update(&mixed_qkv, cache)?
        } else {
            self.causal_conv1d_full(&mixed_qkv, cache)?
        };

        // 4. Split back after conv and reshape to per-head
        let q = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let k = mixed_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v = mixed_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

        let q = q.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

        // 5. Compute beta and g (3D: batch, seq, num_v_heads).
        //
        // Fast path (Metal + FusedAll): pass the raw projection buffer directly
        // to the strided gating kernel, reading b and a by column offsets.
        // Avoids two contiguous-copy Metal dispatches for the non-contiguous
        // narrow views that FusedAll produces.
        let (beta, g) = {
            // Silence "unused variable" on non-Metal builds; proj_all is only
            // consumed inside the #[cfg(feature = "metal")] arm below.
            #[cfg(not(feature = "metal"))]
            let _ = proj_all;
            #[cfg(feature = "metal")]
            {
                if let (Some(proj), true) = (&proj_all, x.device().is_metal()) {
                    if let GdnProjection::FusedAll { z_end, b_end, .. } = &self.projection {
                        let (beta_flat, g_flat) =
                            crate::metal::gdn::fused_gdn_gating_strided_metal(
                                proj,
                                *z_end,
                                *b_end,
                                self.num_v_heads,
                                &self.a_log_f32,
                                &self.dt_bias_f32,
                            )?;
                        let shape = b.shape();
                        (beta_flat.reshape(shape)?, g_flat.reshape(shape)?)
                    } else {
                        self.compute_gating(&b, &a, dtype)?
                    }
                } else {
                    self.compute_gating(&b, &a, dtype)?
                }
            }
            #[cfg(not(feature = "metal"))]
            self.compute_gating(&b, &a, dtype)?
        };

        // 6. If num_v_heads > num_k_heads, repeat_interleave q and k
        let (q, k) = if v_per_group > 1 {
            let q = q
                .unsqueeze(3)?
                .repeat((1, 1, 1, v_per_group, 1))?
                .reshape((batch_size, seq_len, self.num_v_heads, self.head_k_dim))?;
            let k = k
                .unsqueeze(3)?
                .repeat((1, 1, 1, v_per_group, 1))?
                .reshape((batch_size, seq_len, self.num_v_heads, self.head_k_dim))?;
            (q, k)
        } else {
            (q, k)
        };

        // 7. L2-normalize q and k.
        //
        // Metal fast path: fused kernel dispatches one command buffer for both q
        // and k (grid = 2*N groups), saving one Metal dispatch per layer per step.
        let (q, k) = {
            #[cfg(feature = "metal")]
            if q.device().is_metal()
                && matches!(dtype, candle_core::DType::F16 | candle_core::DType::BF16)
            {
                crate::metal::gdn::gdn_l2_norm2_metal(&q, &k, 1e-6_f32)?
            } else {
                (l2_norm(&q, 1e-6)?, l2_norm(&k, 1e-6)?)
            }
            #[cfg(not(feature = "metal"))]
            (l2_norm(&q, 1e-6)?, l2_norm(&k, 1e-6)?)
        };

        // 8. Apply recurrence
        let y = self.apply_recurrence(&q, &k, &v, &g, &beta, batch_size, seq_len, dtype, cache)?;

        cache.seqlen_offset += seq_len;

        // 9. Apply RMSNormGated: flatten to 2D, apply norm with z as gate, reshape back
        let z_shape = z.shape().clone();
        let y = y.reshape(((), self.head_v_dim))?;
        let z = z.reshape(((), self.head_v_dim))?;
        let y = self.norm.forward(&y, &z)?;
        let y = y.reshape(z_shape)?;
        let y = y.reshape((batch_size, seq_len, self.value_dim))?;

        // 10. Output projection
        let original_dtype = x.dtype();
        let mut y_proj = y;
        if let Some(t) = self.out_proj.quantized_act_type() {
            y_proj = y_proj.to_dtype(t)?;
        }
        let mut res = MatMul.qmethod_matmul(&y_proj, &*self.out_proj)?;
        if self.out_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }

    /// Compute beta (sigmoid of b) and g (gating decay from a, A_log, dt_bias).
    ///
    /// Uses cached F32 copies of a_log and dt_bias (set at load time) to avoid
    /// a to_dtype dispatch per call. Called for non-Metal paths and for non-FusedAll
    /// Metal paths; the FusedAll Metal path uses the strided gating kernel in forward().
    fn compute_gating(&self, b: &Tensor, a: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        {
            if b.device().is_cuda() {
                let b_flat = b.contiguous()?.flatten_all()?;
                let a_flat = a.contiguous()?.flatten_all()?;
                let (beta_flat, g_flat) = crate::cuda::gdn::fused_gdn_gating_cuda(
                    &b_flat,
                    &a_flat,
                    &self.a_log_f32,
                    &self.dt_bias_f32,
                )?;
                let shape = b.shape();
                return Ok((beta_flat.reshape(shape)?, g_flat.reshape(shape)?));
            }
        }
        #[cfg(feature = "metal")]
        {
            if b.device().is_metal() {
                let b_flat = b.contiguous()?.flatten_all()?;
                let a_flat = a.contiguous()?.flatten_all()?;
                let (beta_flat, g_flat) = crate::metal::gdn::fused_gdn_gating_metal(
                    &b_flat,
                    &a_flat,
                    &self.a_log_f32,
                    &self.dt_bias_f32,
                )?;
                let shape = b.shape();
                return Ok((beta_flat.reshape(shape)?, g_flat.reshape(shape)?));
            }
        }
        // CPU fallback: use cached F32 tensors to avoid to_dtype allocations.
        let beta = candle_nn::ops::sigmoid(b)?;
        let a_f = a.to_dtype(DType::F32)?;
        let dt_bias_expanded = self.dt_bias_f32.unsqueeze(0)?.unsqueeze(0)?;
        let g = self
            .a_log_f32
            .exp()?
            .neg()?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_mul(&softplus(&a_f.broadcast_add(&dt_bias_expanded)?)?)?
            .to_dtype(dtype)?;
        Ok((beta, g))
    }

    /// Apply recurrence (CUDA or CPU/Metal fallback).
    #[allow(clippy::too_many_arguments, unused_variables)]
    fn apply_recurrence(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        batch_size: usize,
        seq_len: usize,
        dtype: DType,
        cache: &mut GdnLayerCache,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            if q.device().is_cuda() {
                let num_heads = self.num_v_heads;
                let k_head = self.head_k_dim;
                let v_head = self.head_v_dim;
                let scale = 1.0 / (k_head as f64).sqrt();

                // ── Decode fast path (seq_len == 1) ──────────────────────────
                // q/k/v have shape [B, 1, H, D] which shares the same memory
                // layout as [B*H, 1, D] — reshape is a zero-copy view.
                // This eliminates the 2 transpose+contiguous copies per tensor
                // that the prefill path needs.
                let (q_bh, k_bh, v_bh, g_bh, beta_bh) = if seq_len == 1 {
                    let bh = batch_size * num_heads;
                    let q_bh =
                        (q.reshape((bh, 1, k_head))?.to_dtype(DType::F32)? * scale)?
                            .contiguous()?;
                    let k_bh = k
                        .reshape((bh, 1, k_head))?
                        .to_dtype(DType::F32)?
                        .contiguous()?;
                    let v_bh = v
                        .reshape((bh, 1, v_head))?
                        .to_dtype(DType::F32)?
                        .contiguous()?;
                    // g/beta are [B, 1, H] — same zero-copy reshape to [B*H, 1]
                    let g_bh = g
                        .reshape((bh, 1))?
                        .to_dtype(DType::F32)?
                        .contiguous()?;
                    let beta_bh = beta
                        .reshape((bh, 1))?
                        .to_dtype(DType::F32)?
                        .contiguous()?;
                    (q_bh, k_bh, v_bh, g_bh, beta_bh)
                } else {
                    // ── Prefill path (seq_len > 1) ───────────────────────────
                    let q_bh =
                        (q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)? * scale)?
                            .reshape((batch_size * num_heads, seq_len, k_head))?
                            .contiguous()?;
                    let k_bh = k
                        .transpose(1, 2)?
                        .contiguous()?
                        .to_dtype(DType::F32)?
                        .reshape((batch_size * num_heads, seq_len, k_head))?
                        .contiguous()?;
                    let v_bh = v
                        .transpose(1, 2)?
                        .contiguous()?
                        .to_dtype(DType::F32)?
                        .reshape((batch_size * num_heads, seq_len, v_head))?
                        .contiguous()?;
                    let g_bh = g
                        .to_dtype(DType::F32)?
                        .transpose(1, 2)?
                        .contiguous()?
                        .reshape((batch_size * num_heads, seq_len))?
                        .contiguous()?;
                    let beta_bh = beta
                        .to_dtype(DType::F32)?
                        .transpose(1, 2)?
                        .contiguous()?
                        .reshape((batch_size * num_heads, seq_len))?
                        .contiguous()?;
                    (q_bh, k_bh, v_bh, g_bh, beta_bh)
                };

                let mut state_flat = cache
                    .recurrent_state
                    .to_dtype(DType::F32)?
                    .reshape((batch_size * num_heads, k_head, v_head))?
                    .contiguous()?;

                let out_bh = crate::cuda::gdn::gated_delta_rule_recurrence_cuda(
                    &q_bh,
                    &k_bh,
                    &v_bh,
                    &g_bh,
                    &beta_bh,
                    &mut state_flat,
                )?;

                cache.recurrent_state = state_flat
                    .reshape((batch_size, num_heads, k_head, v_head))?
                    .to_dtype(cache.recurrent_state.dtype())?;

                return out_bh
                    .reshape((batch_size, num_heads, seq_len, v_head))?
                    .transpose(1, 2)?
                    .contiguous()?
                    .to_dtype(dtype);
            }
        }

        #[cfg(feature = "metal")]
        {
            if q.device().is_metal() {
                let num_heads = self.num_v_heads;
                let k_head = self.head_k_dim;
                let v_head = self.head_v_dim;
                // Scale is now applied inside the Metal kernel (fused into the q load),
                // so we don't need a separate GPU multiply kernel.
                let q_scale = (1.0 / (k_head as f64).sqrt()) as f32;

                // Tensors arrive as (batch, seq, num_v_heads, head_dim).
                // The kernel expects (batch*heads, seq, head_dim) = [BH, S, D].
                //
                // We use the typed recurrence kernel (gdn_recurrence_typed) which
                // accepts native BF16/F16 input, eliminating 5 to_dtype(F32) GPU
                // kernel dispatches per layer (= 70 per decode step for 14 layers).
                //
                // DECODE FAST PATH (seq_len==1):
                //   (batch, 1, heads, dim) has the same memory layout as
                //   (batch*heads, 1, dim) so we can reshape without transposing.
                //
                // PREFILL PATH (seq_len>1):
                //   Must transpose(1,2) + contiguous() to reorder the strides.
                let model_dtype = q.dtype();
                let use_typed_kernel = matches!(model_dtype, DType::F16 | DType::BF16);

                let (q_bh, k_bh, v_bh, g_bh, beta_bh) = if seq_len == 1 {
                    // Pure-reshape fast path for decode (no GPU kernels at all).
                    let q_bh =
                        q.reshape((batch_size * num_heads, 1, k_head))?;
                    let k_bh =
                        k.reshape((batch_size * num_heads, 1, k_head))?;
                    let v_bh =
                        v.reshape((batch_size * num_heads, 1, v_head))?;
                    let g_bh =
                        g.reshape((batch_size * num_heads, 1))?;
                    let beta_bh =
                        beta.reshape((batch_size * num_heads, 1))?;
                    (q_bh, k_bh, v_bh, g_bh, beta_bh)
                } else if use_typed_kernel {
                    // Prefill: transpose + contiguous only (no to_dtype).
                    let q_bh = q
                        .transpose(1, 2)?
                        .contiguous()?
                        .reshape((batch_size * num_heads, seq_len, k_head))?;
                    let k_bh = k
                        .transpose(1, 2)?
                        .contiguous()?
                        .reshape((batch_size * num_heads, seq_len, k_head))?;
                    let v_bh = v
                        .transpose(1, 2)?
                        .contiguous()?
                        .reshape((batch_size * num_heads, seq_len, v_head))?;
                    let g_bh = g
                        .transpose(1, 2)?
                        .contiguous()?
                        .reshape((batch_size * num_heads, seq_len))?;
                    let beta_bh = beta
                        .transpose(1, 2)?
                        .contiguous()?
                        .reshape((batch_size * num_heads, seq_len))?;
                    (q_bh, k_bh, v_bh, g_bh, beta_bh)
                } else {
                    // F32 fallback: convert + transpose.
                    let q_bh = q
                        .transpose(1, 2)?
                        .contiguous()?
                        .to_dtype(DType::F32)?
                        .reshape((batch_size * num_heads, seq_len, k_head))?;
                    let k_bh = k
                        .transpose(1, 2)?
                        .contiguous()?
                        .to_dtype(DType::F32)?
                        .reshape((batch_size * num_heads, seq_len, k_head))?;
                    let v_bh = v
                        .transpose(1, 2)?
                        .contiguous()?
                        .to_dtype(DType::F32)?
                        .reshape((batch_size * num_heads, seq_len, v_head))?;
                    let g_bh = g
                        .to_dtype(DType::F32)?
                        .transpose(1, 2)?
                        .contiguous()?
                        .reshape((batch_size * num_heads, seq_len))?;
                    let beta_bh = beta
                        .to_dtype(DType::F32)?
                        .transpose(1, 2)?
                        .contiguous()?
                        .reshape((batch_size * num_heads, seq_len))?;
                    (q_bh, k_bh, v_bh, g_bh, beta_bh)
                };

                // recurrent_state is stored in F32 (see GdnLayerCache::new).
                // to_dtype(F32) is a zero-cost Rust clone; reshape is a view.
                let mut state_flat = cache
                    .recurrent_state
                    .to_dtype(DType::F32)?
                    .reshape((batch_size * num_heads, k_head, v_head))?;

                let out_bh = if use_typed_kernel {
                    crate::metal::gdn::gated_delta_rule_recurrence_metal_typed(
                        &q_bh,
                        &k_bh,
                        &v_bh,
                        &g_bh,
                        &beta_bh,
                        &mut state_flat,
                        q_scale,
                    )?
                } else {
                    crate::metal::gdn::gated_delta_rule_recurrence_metal(
                        &q_bh,
                        &k_bh,
                        &v_bh,
                        &g_bh,
                        &beta_bh,
                        &mut state_flat,
                        q_scale,
                    )?
                };

                // state_flat was updated in-place by the Metal kernel.
                // Reshape back and store (already F32, no dtype conversion needed).
                cache.recurrent_state =
                    state_flat.reshape((batch_size, num_heads, k_head, v_head))?;

                return out_bh
                    .reshape((batch_size, num_heads, seq_len, v_head))?
                    .transpose(1, 2)?
                    .contiguous()?
                    .to_dtype(dtype);
            }
        }

        // ── CPU (and any device without a dedicated fused kernel) ────────────
        //
        // Route through the chunk-parallel implementation.  This replaces the
        // original O(seq_len) IndexOp loop with O(n_chunks) sequential state
        // updates whose intra-chunk work is expressed as batch GEMM (BLAS on
        // CPU, cuBLAS / MPS on GPU if this branch is somehow reached there).
        //
        // All tensors are converted to F32 first (matching the existing
        // fallback behaviour) and the scale is fused into q.
        let k_head = self.head_k_dim;
        let v_head = self.head_v_dim;
        let num_heads = self.num_v_heads;
        let scale = 1.0 / (k_head as f64).sqrt();

        // Convert and reshape to the [bh, T, D] layout expected by the chunked
        // recurrence.  For seq_len == 1 the existing Metal fast path already
        // handled it above, so we only reach here for seq_len > 1 on CPU.
        let q_f = (q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)? * scale)?
            .reshape((batch_size * num_heads, seq_len, k_head))?;
        let k_f = k
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(DType::F32)?
            .reshape((batch_size * num_heads, seq_len, k_head))?;
        let v_f = v
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(DType::F32)?
            .reshape((batch_size * num_heads, seq_len, v_head))?;
        let g_f = g
            .to_dtype(DType::F32)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size * num_heads, seq_len))?;
        let beta_f = beta
            .to_dtype(DType::F32)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size * num_heads, seq_len))?;

        let mut state_flat = cache
            .recurrent_state
            .to_dtype(DType::F32)?
            .reshape((batch_size * num_heads, k_head, v_head))?;

        // CHUNK_SIZE = 64 matches llama.cpp's build_delta_net_chunking.
        const CHUNK_SIZE: usize = 64;
        let out_flat = gated_delta_rule_chunked(
            &q_f,
            &k_f,
            &v_f,
            &g_f,
            &beta_f,
            &mut state_flat,
            CHUNK_SIZE,
        )?;

        cache.recurrent_state = state_flat
            .reshape((batch_size, num_heads, k_head, v_head))?
            .to_dtype(cache.recurrent_state.dtype())?;

        out_flat
            .reshape((batch_size, num_heads, seq_len, v_head))?
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(dtype)
    }

    /// Single-step causal conv1d update for decode.
    fn causal_conv1d_update(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (_batch, seq_len, _conv_dim) = x.dims3()?;
        let x_t = x.transpose(1, 2)?.contiguous()?;

        #[cfg(feature = "cuda")]
        if x_t.device().is_cuda() {
            let weight = self
                .conv1d_weight
                .squeeze(1)?
                .to_dtype(x_t.dtype())?
                .contiguous()?;
            let conv_state = cache.conv_state.contiguous()?;
            let (output, new_conv_state) = crate::cuda::gdn::causal_conv1d_cuda(
                &x_t,
                &weight,
                &conv_state,
                self.conv_kernel_size,
                true,
            )?;
            cache.conv_state = new_conv_state;
            return output.transpose(1, 2);
        }

        #[cfg(feature = "metal")]
        if x_t.device().is_metal() {
            let weight = self
                .conv1d_weight
                .squeeze(1)?
                .to_dtype(x_t.dtype())?
                .contiguous()?;
            let conv_state = cache.conv_state.contiguous()?;
            let (output, new_conv_state) = crate::metal::gdn::causal_conv1d_metal(
                &x_t,
                &weight,
                &conv_state,
                self.conv_kernel_size,
                true,
            )?;
            cache.conv_state = new_conv_state;
            return output.transpose(1, 2);
        }

        let state_len = cache.conv_state.dim(2)?;
        let hidden_new = Tensor::cat(&[cache.conv_state.clone(), x_t], 2)?;
        let new_len = hidden_new.dim(2)?;
        cache.conv_state = hidden_new.narrow(2, new_len - state_len, state_len)?;

        let weight = self
            .conv1d_weight
            .squeeze(1)?
            .to_dtype(hidden_new.dtype())?;
        let mut conv_outputs = Vec::with_capacity(seq_len);
        let total_len = hidden_new.dim(2)?;
        for i in (total_len - seq_len)..total_len {
            let window =
                hidden_new.narrow(2, i + 1 - self.conv_kernel_size, self.conv_kernel_size)?;
            let out = (window * weight.unsqueeze(0)?)?.sum(D::Minus1)?;
            conv_outputs.push(out);
        }
        let out = Tensor::stack(&conv_outputs, 2)?;
        let out = candle_nn::ops::silu(&out)?;
        out.transpose(1, 2)
    }

    /// Full sequence causal conv1d for prefill.
    fn causal_conv1d_full(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (batch_size, seq_len, conv_dim) = x.dims3()?;
        let x_t = x.transpose(1, 2)?.contiguous()?;

        #[cfg(feature = "cuda")]
        if x_t.device().is_cuda() {
            let weight = self
                .conv1d_weight
                .squeeze(1)?
                .to_dtype(x_t.dtype())?
                .contiguous()?;
            let (output, new_conv_state) = crate::cuda::gdn::causal_conv1d_cuda(
                &x_t,
                &weight,
                &cache.conv_state,
                self.conv_kernel_size,
                false,
            )?;
            cache.conv_state = new_conv_state;
            return output.transpose(1, 2);
        }

        #[cfg(feature = "metal")]
        if x_t.device().is_metal() {
            let weight = self
                .conv1d_weight
                .squeeze(1)?
                .to_dtype(x_t.dtype())?
                .contiguous()?;
            let (output, new_conv_state) = crate::metal::gdn::causal_conv1d_metal(
                &x_t,
                &weight,
                &cache.conv_state,
                self.conv_kernel_size,
                false,
            )?;
            cache.conv_state = new_conv_state;
            return output.transpose(1, 2);
        }

        // ── CPU causal conv1d ────────────────────────────────────────────────
        //
        // Replacing the original O(seq_len) per-position loop with an
        // O(kernel_size) loop over kernel positions.  For typical GDN models
        // kernel_size = 4, so this is a 4-iteration loop regardless of seq_len.
        //
        // Each iteration slices a length-seq_len window from padded_t and
        // accumulates a weighted sum using broadcast_mul — no per-token
        // Tensor allocation.
        //
        // Save conv_state (last kernel_size positions of the input).
        let pad_width = self.conv_kernel_size.saturating_sub(seq_len);
        cache.conv_state = if pad_width > 0 {
            let zeros =
                Tensor::zeros((batch_size, conv_dim, pad_width), x_t.dtype(), x_t.device())?;
            Tensor::cat(&[zeros, x_t.clone()], 2)?
        } else {
            x_t.narrow(2, seq_len - self.conv_kernel_size, self.conv_kernel_size)?
        };

        // Pad left with (kernel_size - 1) zeros so position 0 looks at a
        // full causal window.
        let padded_t = Tensor::cat(
            &[
                Tensor::zeros(
                    (batch_size, conv_dim, self.conv_kernel_size - 1),
                    x_t.dtype(),
                    x_t.device(),
                )?,
                x_t,
            ],
            2,
        )?;

        // weight: [conv_dim, 1, kernel_size] → squeeze to [conv_dim, kernel_size]
        let weight = self.conv1d_weight.squeeze(1)?.to_dtype(padded_t.dtype())?;

        // Accumulate over K kernel positions (K ≪ seq_len for GDN models).
        // acc[b, c, i] += weight[c, k] * padded_t[b, c, i + k]
        let mut acc: Option<Tensor> = None;
        for k in 0..self.conv_kernel_size {
            // Slice: x_k[b, c, :] = padded_t[b, c, k .. k+seq_len]  [B, C, S]
            let x_k = padded_t.narrow(2, k, seq_len)?;
            // w_k[c] = weight[c, k]  →  broadcast shape [1, C, 1]
            let w_k = weight.narrow(1, k, 1)?.unsqueeze(0)?.unsqueeze(2)?;
            let term = x_k.broadcast_mul(&w_k)?;
            acc = Some(match acc {
                None => term,
                Some(a) => (a + term)?,
            });
        }
        let out = acc.unwrap_or_else(|| {
            Tensor::zeros((batch_size, conv_dim, seq_len), padded_t.dtype(), padded_t.device())
                .expect("zeros")
        });
        let out = candle_nn::ops::silu(&out)?;
        out.transpose(1, 2)
    }
}
