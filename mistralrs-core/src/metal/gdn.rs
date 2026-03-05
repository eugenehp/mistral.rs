//! Metal kernel dispatchers for GatedDeltaNet (GDN) operations.
//!
//! Mirrors src/cuda/gdn.rs but targets Apple Metal (M-series GPUs).

#![allow(clippy::cast_possible_truncation)]

use candle_core::{
    backend::BackendStorage, DType, MetalStorage, Result, Shape, Storage, Tensor,
};
use candle_metal_kernels::metal::{Buffer, ComputeCommandEncoder, ComputePipeline, Device, Library};
use objc2_metal::{MTLCompileOptions, MTLMathMode, MTLSize};
use std::{
    collections::HashMap,
    ffi::c_void,
    sync::{Arc, OnceLock, RwLock},
};

// ============================================================================
// Kernel source + pipeline cache
// ============================================================================

const GDN_METAL_SOURCE: &str = include_str!("gdn.metal");

static GDN_LIBRARY: OnceLock<Library> = OnceLock::new();
static GDN_PIPELINES: OnceLock<RwLock<HashMap<String, ComputePipeline>>> = OnceLock::new();

fn get_library(device: &Device) -> Result<Library> {
    if let Some(lib) = GDN_LIBRARY.get() {
        return Ok(lib.clone());
    }
    let opts = MTLCompileOptions::new();
    opts.setMathMode(MTLMathMode::Safe);
    // candle-metal-kernels' new_library_with_source() calls .unwrap() internally,
    // so a Metal compilation error produces a panic rather than Err. Catch it here
    // and convert to a proper candle Result so the caller's mutex is not poisoned.
    let lib = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        device.new_library_with_source(GDN_METAL_SOURCE, Some(&opts))
    }))
    .map_err(|_| {
        candle_core::Error::Msg(
            "GDN Metal shader compilation panicked – check metal/gdn.metal for syntax errors"
                .into(),
        )
    })?
    .map_err(|e| candle_core::Error::Msg(format!("GDN Metal compile error: {e}")))?;
    Ok(GDN_LIBRARY.get_or_init(|| lib).clone())
}

fn get_pipeline(device: &Device, name: &str) -> Result<ComputePipeline> {
    let pipelines = GDN_PIPELINES.get_or_init(|| RwLock::new(HashMap::new()));
    {
        let guard = pipelines
            .read()
            .map_err(|e| candle_core::Error::Msg(format!("GDN pipeline lock: {e}")))?;
        if let Some(pl) = guard.get(name) {
            return Ok(pl.clone());
        }
    }
    let lib = get_library(device)?;
    let func = lib
        .get_function(name, None)
        .map_err(|e| candle_core::Error::Msg(format!("GDN function '{name}': {e}")))?;
    let pipeline = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        device.new_compute_pipeline_state_with_function(&func)
    }))
    .map_err(|_| {
        candle_core::Error::Msg(format!(
            "GDN Metal pipeline '{name}' panicked during creation"
        ))
    })?
    .map_err(|e| candle_core::Error::Msg(format!("GDN pipeline '{name}': {e}")))?;
    pipelines
        .write()
        .map_err(|e| candle_core::Error::Msg(format!("GDN pipeline lock: {e}")))?
        .insert(name.to_string(), pipeline.clone());
    Ok(pipeline)
}

// ============================================================================
// Small helpers
// ============================================================================

/// Pass a primitive constant to the encoder via set_bytes.
fn set_bytes<T: Sized>(encoder: &ComputeCommandEncoder, index: usize, value: &T) {
    encoder.set_bytes_directly(
        index,
        core::mem::size_of::<T>(),
        (value as *const T).cast::<c_void>(),
    );
}

/// dtype suffix for kernel name selection.
fn dtype_suffix(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("float"),
        DType::F16 => Ok("half"),
        DType::BF16 => Ok("bfloat"),
        other => candle_core::bail!("GDN Metal: unsupported dtype {other:?}"),
    }
}

/// Returns (byte_offset, &Buffer) for a contiguous tensor's Metal storage.
///
/// The caller *must* keep the `RwLockReadGuard` alive (returned as part of
/// the storage) until the encoder is dropped.
fn ms_buf_off<'a>(ms: &'a MetalStorage, t: &Tensor) -> (&'a Buffer, usize) {
    let off = t.layout().start_offset() * ms.dtype().size_in_bytes();
    (ms.buffer(), off)
}

// ============================================================================
// Kernel 1: gated_delta_rule_recurrence_metal
//
// All tensors must be F32 and contiguous.
//   q, k  : [BH, S, K]
//   v     : [BH, S, V]
//   g, beta:[BH, S]
//   state : [BH, K, V]  — mutated in-place; on return holds new state
//
// Returns: output [BH, S, V] in F32.
// ============================================================================

pub fn gated_delta_rule_recurrence_metal(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let (bh, seq_len, k_dim) = q.dims3()?;
    let v_dim = v.dim(2)?;

    // All inputs must be contiguous F32 (caller must ensure this)
    debug_assert_eq!(q.dtype(), DType::F32);

    // Acquire storage guards – must stay alive until encoder is dropped
    let q_sg = q.storage_and_layout().0;
    let k_sg = k.storage_and_layout().0;
    let v_sg = v.storage_and_layout().0;
    let g_sg = g.storage_and_layout().0;
    let beta_sg = beta.storage_and_layout().0;
    let state_sg = state.storage_and_layout().0;

    let Storage::Metal(q_ms) = &*q_sg else {
        candle_core::bail!("gdn_recurrence: q must be Metal")
    };
    let Storage::Metal(k_ms) = &*k_sg else {
        candle_core::bail!("gdn_recurrence: k must be Metal")
    };
    let Storage::Metal(v_ms) = &*v_sg else {
        candle_core::bail!("gdn_recurrence: v must be Metal")
    };
    let Storage::Metal(g_ms) = &*g_sg else {
        candle_core::bail!("gdn_recurrence: g must be Metal")
    };
    let Storage::Metal(beta_ms) = &*beta_sg else {
        candle_core::bail!("gdn_recurrence: beta must be Metal")
    };
    let Storage::Metal(state_ms) = &*state_sg else {
        candle_core::bail!("gdn_recurrence: state must be Metal")
    };

    let device = q_ms.device();
    let raw_device = device.metal_device();

    let (q_buf, q_off) = ms_buf_off(q_ms, q);
    let (k_buf, k_off) = ms_buf_off(k_ms, k);
    let (v_buf, v_off) = ms_buf_off(v_ms, v);
    let (g_buf, g_off) = ms_buf_off(g_ms, g);
    let (beta_buf, beta_off) = ms_buf_off(beta_ms, beta);
    let (state_buf, state_off) = ms_buf_off(state_ms, state);

    // Allocate output
    let out_arc: Arc<Buffer> =
        device.new_buffer(bh * seq_len * v_dim, DType::F32, "gdn_recurrence_out")?;

    let encoder = device.command_encoder()?;
    encoder.set_label("gdn_recurrence");

    let pipeline = get_pipeline(raw_device, "gdn_recurrence")?;
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(q_buf), q_off);
    encoder.set_buffer(1, Some(k_buf), k_off);
    encoder.set_buffer(2, Some(v_buf), v_off);
    encoder.set_buffer(3, Some(g_buf), g_off);
    encoder.set_buffer(4, Some(beta_buf), beta_off);
    encoder.set_buffer(5, Some(state_buf), state_off); // read + written in-place
    encoder.set_buffer(6, Some(&*out_arc), 0);

    let sl = seq_len as u32;
    let kd = k_dim as u32;
    let vd = v_dim as u32;
    set_bytes(&encoder, 7, &sl);
    set_bytes(&encoder, 8, &kd);
    set_bytes(&encoder, 9, &vd);

    // Shared memory: 2 * k_dim floats
    encoder.set_threadgroup_memory_length(0, 2 * k_dim * core::mem::size_of::<f32>());

    const BV: usize = 64;
    let v_tiles = v_dim.div_ceil(BV);
    encoder.dispatch_thread_groups(
        MTLSize { width: v_tiles, height: bh, depth: 1 },
        MTLSize { width: BV, height: 1, depth: 1 },
    );

    drop(encoder);

    // state buffer was modified in-place; the caller's `state` tensor still
    // owns that buffer – no need to create a new tensor for it.

    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(
            out_arc,
            device.clone(),
            bh * seq_len * v_dim,
            DType::F32,
        )),
        Shape::from_dims(&[bh, seq_len, v_dim]),
    )))
}

// ============================================================================
// Kernel 2 & 3: causal_conv1d_metal
//
// is_update = true  (decode, single new token):
//   x          : [B, conv_dim]  (may have trailing size-1 dim)
//   weight     : [conv_dim, kernel_size]
//   conv_state : [B, conv_dim, kernel_size] — mutated in-place
//   Returns    : (output [B, conv_dim, 1], new_conv_state [B, conv_dim, kernel_size])
//
// is_update = false (prefill, full sequence):
//   x          : [B, conv_dim, S]
//   Returns    : (output [B, conv_dim, S], new_conv_state [B, conv_dim, kernel_size])
// ============================================================================

pub fn causal_conv1d_metal(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
    kernel_size: usize,
    is_update: bool,
) -> Result<(Tensor, Tensor)> {
    let dtype = x.dtype();
    let suffix = dtype_suffix(dtype)?;

    const THREADS: usize = 256;

    let x_sg = x.storage_and_layout().0;
    let w_sg = weight.storage_and_layout().0;

    let Storage::Metal(x_ms) = &*x_sg else {
        candle_core::bail!("causal_conv1d_metal: x must be Metal")
    };
    let Storage::Metal(w_ms) = &*w_sg else {
        candle_core::bail!("causal_conv1d_metal: weight must be Metal")
    };

    let device = x_ms.device();
    let raw_device = device.metal_device();

    let (x_buf, x_off) = ms_buf_off(x_ms, x);
    let (w_buf, w_off) = ms_buf_off(w_ms, weight);

    if is_update {
        // x may be [B, C] or [B, C, 1]
        let dims = x.dims();
        let (batch_size, conv_dim) = match dims.len() {
            2 => (dims[0], dims[1]),
            3 => (dims[0], dims[1]),
            _ => candle_core::bail!("causal_conv1d_update: bad x dims {:?}", dims),
        };

        let cs_sg = conv_state.storage_and_layout().0;
        let Storage::Metal(cs_ms) = &*cs_sg else {
            candle_core::bail!("causal_conv1d_metal: conv_state must be Metal")
        };
        let (cs_buf, cs_off) = ms_buf_off(cs_ms, conv_state);

        let out_arc: Arc<Buffer> =
            device.new_buffer(batch_size * conv_dim, dtype, "gdn_conv1d_update_out")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("gdn_conv1d_update");

        let name = format!("gdn_conv1d_update_{suffix}");
        let pipeline = get_pipeline(raw_device, &name)?;
        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_buffer(0, Some(x_buf), x_off);
        encoder.set_buffer(1, Some(w_buf), w_off);
        encoder.set_buffer(2, Some(cs_buf), cs_off); // in-place
        encoder.set_buffer(3, Some(&*out_arc), 0);

        let cd = conv_dim as u32;
        let ks = kernel_size as u32;
        set_bytes(&encoder, 4, &cd);
        set_bytes(&encoder, 5, &ks);

        let ch_groups = conv_dim.div_ceil(THREADS);
        encoder.dispatch_thread_groups(
            MTLSize { width: ch_groups, height: batch_size, depth: 1 },
            MTLSize { width: THREADS, height: 1, depth: 1 },
        );

        drop(encoder);

        // conv_state buffer was updated in-place by the kernel.
        // Return a clone of the existing tensor handle (same Arc<Buffer>,
        // now containing updated data on the GPU timeline).
        let new_cs = conv_state.clone();

        let output = Tensor::from((
            Storage::Metal(MetalStorage::new(
                out_arc,
                device.clone(),
                batch_size * conv_dim,
                dtype,
            )),
            Shape::from_dims(&[batch_size, conv_dim, 1]),
        ));

        Ok((output, new_cs))
    } else {
        // Full sequence: x = [B, conv_dim, S]
        let (batch_size, conv_dim, seq_len) = x.dims3()?;

        let out_arc: Arc<Buffer> =
            device.new_buffer(batch_size * conv_dim * seq_len, dtype, "gdn_conv1d_full_out")?;
        let cs_arc: Arc<Buffer> =
            device.new_buffer(batch_size * conv_dim * kernel_size, dtype, "gdn_conv_state_new")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("gdn_conv1d_full");

        // Kernel 3a: causal_conv1d_full
        {
            let name = format!("gdn_conv1d_full_{suffix}");
            let pipeline = get_pipeline(raw_device, &name)?;
            encoder.set_compute_pipeline_state(&pipeline);

            encoder.set_buffer(0, Some(x_buf), x_off);
            encoder.set_buffer(1, Some(w_buf), w_off);
            encoder.set_buffer(2, Some(&*out_arc), 0);

            let cd = conv_dim as u32;
            let sl = seq_len as u32;
            let ks = kernel_size as u32;
            set_bytes(&encoder, 3, &cd);
            set_bytes(&encoder, 4, &sl);
            set_bytes(&encoder, 5, &ks);

            let ch_groups = conv_dim.div_ceil(THREADS);
            encoder.dispatch_thread_groups(
                MTLSize { width: ch_groups, height: seq_len, depth: batch_size },
                MTLSize { width: THREADS, height: 1, depth: 1 },
            );
        }

        // Kernel 3b: save_conv_state
        {
            let name = format!("gdn_save_conv_state_{suffix}");
            let pipeline = get_pipeline(raw_device, &name)?;
            encoder.set_compute_pipeline_state(&pipeline);

            encoder.set_buffer(0, Some(x_buf), x_off);
            encoder.set_buffer(1, Some(&*cs_arc), 0);

            let cd = conv_dim as u32;
            let sl = seq_len as u32;
            let ks = kernel_size as u32;
            set_bytes(&encoder, 2, &cd);
            set_bytes(&encoder, 3, &sl);
            set_bytes(&encoder, 4, &ks);

            let ch_groups = conv_dim.div_ceil(THREADS);
            encoder.dispatch_thread_groups(
                MTLSize { width: ch_groups, height: batch_size, depth: 1 },
                MTLSize { width: THREADS, height: 1, depth: 1 },
            );
        }

        drop(encoder);

        let output = Tensor::from((
            Storage::Metal(MetalStorage::new(
                out_arc,
                device.clone(),
                batch_size * conv_dim * seq_len,
                dtype,
            )),
            Shape::from_dims(&[batch_size, conv_dim, seq_len]),
        ));

        let new_cs = Tensor::from((
            Storage::Metal(MetalStorage::new(
                cs_arc,
                device.clone(),
                batch_size * conv_dim * kernel_size,
                dtype,
            )),
            Shape::from_dims(&[batch_size, conv_dim, kernel_size]),
        ));

        Ok((output, new_cs))
    }
}

// ============================================================================
// Kernel 4: fused_gdn_gating_metal
//
// b, a      : [total_elements] in F16/BF16 (contiguous, flat)
// a_log,
// dt_bias   : [num_heads] in F32
//
// Returns: (beta, g) in same dtype as b, with same shape as b.
// ============================================================================

pub fn fused_gdn_gating_metal(
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let dtype = b.dtype();
    let suffix = dtype_suffix(dtype)?;

    let total = b.elem_count();
    let num_heads = a_log.elem_count();
    let shape = b.shape().clone();

    let b_sg = b.storage_and_layout().0;
    let a_sg = a.storage_and_layout().0;
    let alog_sg = a_log.storage_and_layout().0;
    let dtb_sg = dt_bias.storage_and_layout().0;

    let Storage::Metal(b_ms) = &*b_sg else {
        candle_core::bail!("fused_gdn_gating_metal: b must be Metal")
    };
    let Storage::Metal(a_ms) = &*a_sg else {
        candle_core::bail!("fused_gdn_gating_metal: a must be Metal")
    };
    let Storage::Metal(alog_ms) = &*alog_sg else {
        candle_core::bail!("fused_gdn_gating_metal: a_log must be Metal")
    };
    let Storage::Metal(dtb_ms) = &*dtb_sg else {
        candle_core::bail!("fused_gdn_gating_metal: dt_bias must be Metal")
    };

    let device = b_ms.device();
    let raw_device = device.metal_device();

    let (b_buf, b_off) = ms_buf_off(b_ms, b);
    let (a_buf, a_off) = ms_buf_off(a_ms, a);
    let (alog_buf, alog_off) = ms_buf_off(alog_ms, a_log);
    let (dtb_buf, dtb_off) = ms_buf_off(dtb_ms, dt_bias);

    let beta_arc: Arc<Buffer> = device.new_buffer(total, dtype, "gdn_beta_out")?;
    let g_arc: Arc<Buffer> = device.new_buffer(total, dtype, "gdn_g_out")?;

    let encoder = device.command_encoder()?;
    encoder.set_label("gdn_gating");

    let name = format!("gdn_gating_{suffix}");
    let pipeline = get_pipeline(raw_device, &name)?;
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(b_buf), b_off);
    encoder.set_buffer(1, Some(a_buf), a_off);
    encoder.set_buffer(2, Some(alog_buf), alog_off);
    encoder.set_buffer(3, Some(dtb_buf), dtb_off);
    encoder.set_buffer(4, Some(&*beta_arc), 0);
    encoder.set_buffer(5, Some(&*g_arc), 0);

    let nh = num_heads as u32;
    let te = total as u32;
    set_bytes(&encoder, 6, &nh);
    set_bytes(&encoder, 7, &te);

    const THREADS: usize = 256;
    let groups = total.div_ceil(THREADS);
    encoder.dispatch_thread_groups(
        MTLSize { width: groups, height: 1, depth: 1 },
        MTLSize { width: THREADS, height: 1, depth: 1 },
    );

    drop(encoder);

    let beta = Tensor::from((
        Storage::Metal(MetalStorage::new(
            beta_arc,
            device.clone(),
            total,
            dtype,
        )),
        shape.clone(),
    ));
    let g = Tensor::from((
        Storage::Metal(MetalStorage::new(g_arc, device.clone(), total, dtype)),
        shape,
    ));

    Ok((beta, g))
}
