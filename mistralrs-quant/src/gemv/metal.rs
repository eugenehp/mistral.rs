/// Metal GEMV fast path for single-token decode.
///
/// Uses the same `OnceLock` library/pipeline caching pattern as the rest of
/// mistralrs-quant's Metal kernel infrastructure.

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use candle_core::{backend::BackendStorage, DType, Result, Shape, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputeCommandEncoder, ComputePipeline, Library};
use objc2_metal::{MTLCompileOptions, MTLSize};

const GEMV_SOURCE: &str = include_str!("gemv.metal");

static GEMV_LIB: OnceLock<Library> = OnceLock::new();
static GEMV_PIPELINES: OnceLock<RwLock<HashMap<String, ComputePipeline>>> = OnceLock::new();

fn get_library(device: &candle_metal_kernels::metal::Device) -> Result<Library> {
    if let Some(lib) = GEMV_LIB.get() {
        return Ok(lib.clone());
    }
    let opts = MTLCompileOptions::new();
    let lib = device
        .new_library_with_source(GEMV_SOURCE, Some(&opts))
        .map_err(|e| candle_core::Error::Msg(format!("GEMV Metal compile: {e}")))?;
    Ok(GEMV_LIB.get_or_init(|| lib).clone())
}

fn get_pipeline(device: &candle_metal_kernels::metal::Device, name: &str) -> Result<ComputePipeline> {
    let map = GEMV_PIPELINES.get_or_init(|| RwLock::new(HashMap::new()));
    {
        let g = map.read().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        if let Some(p) = g.get(name) {
            return Ok(p.clone());
        }
    }
    let lib = get_library(device)?;
    let func = lib
        .get_function(name, None)
        .map_err(|e| candle_core::Error::Msg(format!("GEMV kernel '{name}': {e}")))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| candle_core::Error::Msg(format!("GEMV pipeline: {e}")))?;
    map.write()
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?
        .insert(name.to_string(), pipeline.clone());
    Ok(pipeline)
}

fn ms_buf_off(ms: &candle_core::MetalStorage, t: &Tensor) -> (Buffer, usize) {
    let (_storage, layout) = t.storage_and_layout();
    let off = layout.start_offset() * ms.dtype().size_in_bytes();
    (ms.buffer().clone(), off)
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Returns `true` when `metal_gemv` should replace Candle's MLX GEMM.
pub fn should_use_metal_gemv(x: &Tensor, w: &Tensor) -> bool {
    use candle_core::DType::{BF16, F16, F32};
    if !x.device().is_metal() { return false; }
    let dtype = x.dtype();
    if !matches!(dtype, BF16 | F16 | F32) { return false; }
    if dtype != w.dtype() { return false; }
    let x_dims = x.dims();
    let m: usize = x_dims[..x_dims.len().saturating_sub(1)].iter().product();
    if m == 0 || m > 8 { return false; }
    let k = x.dim(x.rank() - 1).unwrap_or(0);
    if k % 2 != 0 { return false; }
    w.dim(w.rank() - 1).map(|wk| wk == k).unwrap_or(false)
}

/// `y = x @ W^T`   (no bias — caller adds bias if needed)
pub fn metal_gemv(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let dtype  = x.dtype();
    let x_dims = x.dims();
    let m: usize = x_dims[..x_dims.len().saturating_sub(1)]
        .iter()
        .product::<usize>()
        .max(1);
    let k       = x.dim(x.rank() - 1)?;
    let (n_out, n_in) = w.dims2()?;
    debug_assert_eq!(n_in, k);

    let x_c = x.contiguous()?;
    let w_c = w.contiguous()?;

    let x_sg = x_c.storage_and_layout().0;
    let w_sg = w_c.storage_and_layout().0;
    let Storage::Metal(x_ms) = &*x_sg else {
        candle_core::bail!("metal_gemv: x must be Metal")
    };
    let Storage::Metal(w_ms) = &*w_sg else {
        candle_core::bail!("metal_gemv: w must be Metal")
    };

    let device     = x_ms.device();
    let raw_device = device.metal_device();

    let (x_buf, x_off) = ms_buf_off(x_ms, &x_c);
    let (w_buf, w_off) = ms_buf_off(w_ms, &w_c);

    let out_buf = device.new_buffer(m * n_out, dtype, "gemv_out")?;

    let encoder = device.command_encoder()?;
    encoder.set_label("metal_gemv");

    let suffix = match dtype {
        DType::F32  => "f32",
        DType::F16  => "f16",
        DType::BF16 => "bf16",
        d => candle_core::bail!("metal_gemv: unsupported dtype {d:?}"),
    };
    let pipeline = get_pipeline(raw_device, &format!("gemv_{suffix}"))?;

    // `encoder` is a `candle_metal_kernels::metal::ComputeCommandEncoder`
    // Use the methods available in that type.
    let enc: &ComputeCommandEncoder = encoder.as_ref();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&w_buf), w_off);
    enc.set_buffer(1, Some(&x_buf), x_off);
    enc.set_buffer(2, Some(&*out_buf), 0);

    let no = n_out as u32;
    let ni = n_in  as u32;
    let mb = m     as u32;
    enc.set_bytes(0 + 3, &no);   // buffer(3)
    enc.set_bytes(1 + 3, &ni);   // buffer(4)
    enc.set_bytes(2 + 3, &mb);   // buffer(5)

    const TG_K: usize = 32;
    const TG_N: usize = 4;
    enc.set_threadgroup_memory_length(0, TG_K * TG_N * core::mem::size_of::<f32>());

    let n_groups = n_out.div_ceil(TG_N);
    enc.dispatch_thread_groups(
        MTLSize { width: m,        height: n_groups, depth: 1 },
        MTLSize { width: TG_K,     height: TG_N,     depth: 1 },
    );
    drop(encoder);

    let mut out_shape = x_dims[..x_dims.len() - 1].to_vec();
    out_shape.push(n_out);

    Ok(Tensor::from((
        Storage::Metal(candle_core::MetalStorage::new(
            (*out_buf).clone().into(),
            device.clone(),
            m * n_out,
            dtype,
        )),
        Shape::from_dims(&out_shape),
    )))
}
