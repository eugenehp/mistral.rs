#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Qwen3.5 dense text model (`Qwen3_5ForCausalLM`).
//!
//! Architecture: hybrid full-attention + GatedDeltaNet linear-attention layers,
//! with a dense MLP (no MoE).  Layer types are given explicitly in the config
//! via the `layer_types` field (same format as the Qwen3.5-VL text backbone).
//!
//! For the MoE text variant (`Qwen3_5MoeForCausalLM`) a separate loader
//! (`Qwen3_5MoeLoader`) reuses the same `Model` with `is_moe = true`.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Embedding;
use mistralrs_quant::{
    QuantMethod, QuantizedConfig, ReplicatedLayer,
    ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::{DeviceMappedMask, DeviceMapper},
    kv_cache::{HybridCache, HybridCacheConfig, HybridLayerType},
    layers::{embedding, CausalMasker, GemmaRmsNorm, MatMul, RotaryEmbedding},
    layers_masker::PastKvLenCache,
    models::{
        deltanet::{DeltaNetConfig, GatedDeltaNet, GdnLayerCache, GdnProjection},
        qwen3_next::{FullAttention, Mlp, SparseMoeBlock},
    },
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        NormalModel,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalLoadingMetadata,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

serde_default_fn!(f64, default_partial_rotary_factor, 0.25);
serde_default_fn!(bool, default_norm_topk_prob, true);

// ====================== Config ======================

/// Optional nested rope-parameters block (same as Qwen3.5-VL).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RopeParameters {
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
}

/// Top-level config for `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,

    // RoPE — may be a flat field or nested under rope_parameters.
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,

    // Explicit layer-type list: "full_attention" | "attention" or anything else → linear.
    pub layer_types: Vec<String>,

    // GDN (GatedDeltaNet) fields.
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,

    // Dense MLP.
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default)]
    pub hidden_act: Option<crate::layers::Activation>,

    // MoE fields (only used when `is_moe = true` in the loader).
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,

    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
}

impl Config {
    /// Effective RoPE theta (flat field wins over nested).
    pub fn rope_theta(&self) -> f64 {
        self.rope_theta
            .or_else(|| self.rope_parameters.as_ref().and_then(|p| p.rope_theta))
            .unwrap_or(10_000_000.0)
    }

    /// Partial rotary factor (flat field wins over nested).
    pub fn partial_rotary_factor(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map(|p| p.partial_rotary_factor)
            .unwrap_or(self.partial_rotary_factor)
    }

    /// Activation function, defaulting to SiLU.
    pub fn activation(&self) -> crate::layers::Activation {
        self.hidden_act
            .unwrap_or(crate::layers::Activation::Silu)
    }

    /// Dense MLP intermediate size.
    pub fn dense_intermediate_size(&self) -> usize {
        self.intermediate_size
            .unwrap_or(self.shared_expert_intermediate_size)
    }

    /// Parse layer types from the explicit string list.
    pub fn layer_types_parsed(&self) -> Vec<LayerType> {
        self.layer_types
            .iter()
            .map(|s| match s.as_str() {
                "full_attention" | "attention" => LayerType::FullAttention,
                _ => LayerType::LinearAttention,
            })
            .collect()
    }

    /// conv dim for GDN = key_dim * 2 + value_dim.
    pub fn linear_conv_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim * 2
            + self.linear_num_value_heads * self.linear_value_head_dim
    }
}

// ===== DeltaNetConfig impl =====

struct ConfigAdapter<'a>(&'a Config);

impl<'a> DeltaNetConfig for ConfigAdapter<'a> {
    fn hidden_size(&self) -> usize {
        self.0.hidden_size
    }
    fn rms_norm_eps(&self) -> f64 {
        self.0.rms_norm_eps
    }
    fn linear_num_key_heads(&self) -> usize {
        self.0.linear_num_key_heads
    }
    fn linear_num_value_heads(&self) -> usize {
        self.0.linear_num_value_heads
    }
    fn linear_key_head_dim(&self) -> usize {
        self.0.linear_key_head_dim
    }
    fn linear_value_head_dim(&self) -> usize {
        self.0.linear_value_head_dim
    }
    fn linear_conv_kernel_dim(&self) -> usize {
        self.0.linear_conv_kernel_dim
    }
    fn quantization_config(&self) -> &Option<QuantizedConfig> {
        &self.0.quantization_config
    }
}

// ====================== Layer types ======================

#[derive(Debug, Clone)]
pub enum LayerType {
    FullAttention,
    LinearAttention,
}

// ====================== Feed-forward ======================

enum FeedForward {
    Dense(Mlp),
    MoE(SparseMoeBlock),
}

impl FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            FeedForward::Dense(mlp) => mlp.forward(x),
            FeedForward::MoE(moe) => moe.forward(x),
        }
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        match self {
            FeedForward::Dense(mlp) => mlp.get_isq_layers(),
            FeedForward::MoE(moe) => moe.get_isq_layers(),
        }
    }
}

// ====================== Decoder layer ======================

pub(crate) enum LayerImplVariant {
    FullAttention(FullAttention),
    LinearAttention(GatedDeltaNet),
}

struct DecoderLayer {
    layer_impl: LayerImplVariant,
    input_layernorm: GemmaRmsNorm,
    post_attention_layernorm: GemmaRmsNorm,
    ffn: FeedForward,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn forward_attention(
        &self,
        x: &Tensor,
        attention_mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let attn = match &self.layer_impl {
            LayerImplVariant::FullAttention(attn) => attn,
            LayerImplVariant::LinearAttention(_) => {
                candle_core::bail!("forward_attention called on a LinearAttention layer")
            }
        };
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let attn_out = attn.forward(&x, attention_mask, seqlen_offsets, kv_cache, metadata, flash_params)?;
        let x = (attn_out + residual)?;
        let residual = &x;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let ffn_out = self.ffn.forward(&normed)?;
        ffn_out + residual
    }

    fn forward_linear(&self, x: &Tensor, gdn_cache: &mut GdnLayerCache) -> Result<Tensor> {
        let gdn = match &self.layer_impl {
            LayerImplVariant::LinearAttention(gdn) => gdn,
            LayerImplVariant::FullAttention(_) => {
                candle_core::bail!("forward_linear called on a FullAttention layer")
            }
        };
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let gdn_out = gdn.forward(&x, gdn_cache)?;
        let x = (gdn_out + residual)?;
        let residual = &x;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let ffn_out = self.ffn.forward(&normed)?;
        ffn_out + residual
    }
}

// ====================== Local hybrid cache ======================

enum LocalLayerCache {
    Attention(KvCache),
    LinearAttention(GdnLayerCache),
}

struct LocalHybridCache {
    caches: Vec<LocalLayerCache>,
}

impl LocalHybridCache {
    fn new(
        layer_types: &[LayerType],
        cfg: &Config,
        device: &Device,
        dtype: DType,
        world_size: usize,
    ) -> Result<Self> {
        let adapter = ConfigAdapter(cfg);
        let mut caches = Vec::with_capacity(layer_types.len());
        for lt in layer_types {
            match lt {
                LayerType::FullAttention => {
                    caches.push(LocalLayerCache::Attention(KvCache::new_normal(
                        2,
                        cfg.max_position_embeddings,
                        HybridCache::CACHE_GROW_SIZE,
                    )));
                }
                LayerType::LinearAttention => {
                    caches.push(LocalLayerCache::LinearAttention(GdnLayerCache::new(
                        &adapter,
                        dtype,
                        device,
                        world_size,
                    )?));
                }
            }
        }
        Ok(Self { caches })
    }

    fn seqlen(&self) -> usize {
        for cache in &self.caches {
            if let LocalLayerCache::Attention(kv) = cache {
                return kv.current_seq_len();
            }
        }
        0
    }
}

impl PastKvLenCache for LocalHybridCache {
    fn get_past_kv_len(&self) -> Result<usize> {
        Ok(self.seqlen())
    }
}

// ====================== Top-level Model ======================

pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    #[allow(dead_code)]
    layer_types: Vec<LayerType>,
    norm: GemmaRmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    local_cache: Arc<Mutex<LocalHybridCache>>,
    kv_cache: EitherCache,
    device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
    num_attention_heads: usize,
    max_seq_len: usize,
}

impl Model {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        is_moe: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let vb_lm_head = vb.pp("lm_head");

        if let Some(ref quant_cfg) = &config.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb_m)
            );
        }

        let mapper = normal_loading_metadata.mapper;

        let embed_tokens = embedding(
            config.vocab_size,
            config.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &config.quantization_config,
        )?;

        let lm_head = if !config.tie_word_embeddings {
            ReplicatedLayer::new(
                config.hidden_size,
                config.vocab_size,
                &config.quantization_config,
                false,
                mapper.set_nm_device(vb_lm_head, normal_loading_metadata.loading_isq),
            )?
        } else {
            ReplicatedLayer::from_linear(candle_nn::Linear::new(
                mapper.cast_nm_device(
                    embed_tokens.embeddings(),
                    normal_loading_metadata.loading_isq,
                )?,
                None,
            ))?
        };

        let norm = GemmaRmsNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;

        let layer_types = config.layer_types_parsed();
        let adapter = ConfigAdapter(config);
        let rot_dim =
            (config.head_dim as f64 * config.partial_rotary_factor()) as usize;

        // Build RotaryEmbedding for full-attention layers.
        let mut ropes = HashMap::new();
        for (i, lt) in layer_types.iter().enumerate().take(config.num_hidden_layers) {
            if matches!(lt, LayerType::FullAttention) {
                let device = mapper
                    .device_for(i, false)
                    .unwrap_or(&normal_loading_metadata.real_device);
                if let std::collections::hash_map::Entry::Vacant(e) =
                    ropes.entry(device.location())
                {
                    let rope = RotaryEmbedding::new_partial(
                        config.rope_theta() as f32,
                        rot_dim,
                        config.max_position_embeddings,
                        device,
                        is_gptx,
                        vb_m.dtype(),
                    )?;
                    e.insert(Arc::new(rope));
                }
            }
        }

        let num_full = layer_types
            .iter()
            .filter(|t| matches!(t, LayerType::FullAttention))
            .count();
        let num_linear = layer_types
            .iter()
            .filter(|t| matches!(t, LayerType::LinearAttention))
            .count();
        if is_moe {
            tracing::info!(
                "Qwen3.5-MoE: {} full attention layers, {} linear attention (GDN) layers",
                num_full,
                num_linear
            );
        } else {
            tracing::info!(
                "Qwen3.5: {} full attention layers, {} linear attention (GDN) layers",
                num_full,
                num_linear
            );
        }

        // Build a dummy qwen3_next::Config for reusing FullAttention / SparseMoeBlock.
        let dummy_cfg_for_layer = |_i: usize| -> crate::models::qwen3_next::Config {
            let dummy_moe_intermediate_size = if is_moe { config.moe_intermediate_size } else { 0 };
            let dummy_shared_expert_intermediate_size =
                if is_moe { config.shared_expert_intermediate_size } else { 0 };
            let dummy_num_experts = if is_moe { config.num_experts } else { 0 };
            let dummy_num_experts_per_tok = if is_moe { config.num_experts_per_tok } else { 0 };
            crate::models::qwen3_next::Config {
                vocab_size: config.vocab_size,
                hidden_size: config.hidden_size,
                intermediate_size: config.intermediate_size.unwrap_or(0),
                num_hidden_layers: config.num_hidden_layers,
                num_attention_heads: config.num_attention_heads,
                num_key_value_heads: config.num_key_value_heads,
                hidden_act: config.activation(),
                max_position_embeddings: config.max_position_embeddings,
                rms_norm_eps: config.rms_norm_eps,
                rope_theta: config.rope_theta(),
                head_dim: config.head_dim,
                partial_rotary_factor: config.partial_rotary_factor(),
                linear_conv_kernel_dim: config.linear_conv_kernel_dim,
                linear_key_head_dim: config.linear_key_head_dim,
                linear_value_head_dim: config.linear_value_head_dim,
                linear_num_key_heads: config.linear_num_key_heads,
                linear_num_value_heads: config.linear_num_value_heads,
                decoder_sparse_step: 1,
                moe_intermediate_size: dummy_moe_intermediate_size,
                shared_expert_intermediate_size: dummy_shared_expert_intermediate_size,
                num_experts_per_tok: dummy_num_experts_per_tok,
                num_experts: dummy_num_experts,
                norm_topk_prob: config.norm_topk_prob,
                mlp_only_layers: config.mlp_only_layers.clone(),
                full_attention_interval: 4,
                tie_word_embeddings: config.tie_word_embeddings,
                quantization_config: config.quantization_config.clone(),
            }
        };

        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in NiceProgressBar::<_, 'b'>(
            0..config.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        ) {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let comm = mapper.get_comm_for(i)?;
            let vb_layer = vb_l.pp(i);
            let dummy_cfg = dummy_cfg_for_layer(i);

            let layer_impl = match &layer_types[i] {
                LayerType::FullAttention => {
                    let rotary_emb = ropes
                        .get(&device.location())
                        .expect("No RoPE for device location")
                        .clone();
                    let paged_attn = match &attention_mechanism {
                        AttentionImplementation::Eager => None,
                        AttentionImplementation::PagedAttention => {
                            Some(PagedAttention::new(config.head_dim, device, None)?)
                        }
                    };
                    LayerImplVariant::FullAttention(FullAttention::load(
                        vb_layer.clone(),
                        &dummy_cfg,
                        &*mapper,
                        i,
                        normal_loading_metadata.loading_isq,
                        rotary_emb,
                        paged_attn,
                        &comm,
                    )?)
                }
                LayerType::LinearAttention => {
                    LayerImplVariant::LinearAttention(GatedDeltaNet::load_qwen3_5(
                        vb_layer.clone(),
                        &adapter,
                        &*mapper,
                        i,
                        normal_loading_metadata.loading_isq,
                        &comm,
                    )?)
                }
            };

            let input_layernorm = GemmaRmsNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                mapper.set_device(i, vb_layer.pp("input_layernorm"), false),
            )?;
            let post_attention_layernorm = GemmaRmsNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                mapper.set_device(i, vb_layer.pp("post_attention_layernorm"), false),
            )?;

            let is_moe_layer = is_moe && !config.mlp_only_layers.contains(&i);
            let ffn = if is_moe_layer {
                FeedForward::MoE(SparseMoeBlock::new(
                    &dummy_cfg,
                    mapper.set_device(i, vb_layer.pp("mlp"), normal_loading_metadata.loading_isq),
                    &*mapper,
                    i,
                    normal_loading_metadata.loading_isq,
                    &comm,
                    normal_loading_metadata.real_device.clone(),
                    true,
                )?)
            } else {
                FeedForward::Dense(Mlp::new(
                    mapper.set_device(i, vb_layer.pp("mlp"), normal_loading_metadata.loading_isq),
                    config.hidden_size,
                    config.dense_intermediate_size(),
                    &config.quantization_config,
                    config.activation(),
                    &comm,
                )?)
            };

            layers.push(DecoderLayer {
                layer_impl,
                input_layernorm,
                post_attention_layernorm,
                ffn,
            });
        }

        // Build hybrid cache.
        let local_cache = Arc::new(Mutex::new(LocalHybridCache::new(
            &layer_types,
            config,
            &normal_loading_metadata.real_device,
            vb_m.dtype(),
            mapper.get_comm_for(0)?.world_size(),
        )?));

        let pipeline_layer_types: Vec<HybridLayerType> = layer_types
            .iter()
            .map(|lt| match lt {
                LayerType::FullAttention => HybridLayerType::Attention,
                LayerType::LinearAttention => HybridLayerType::Mamba,
            })
            .collect();

        let hybrid_cache_config = HybridCacheConfig {
            layer_types: pipeline_layer_types,
            max_seq_len: config.max_position_embeddings,
            max_num_seqs: 1,
            mamba_conv_dim: config.linear_conv_dim(),
            mamba_d_conv: config.linear_conv_kernel_dim,
            mamba_n_heads: config.linear_num_value_heads,
            mamba_head_dim: config.linear_key_head_dim,
            mamba_d_state: config.linear_value_head_dim,
        };

        let pipeline_cache = Arc::new(Mutex::new(
            HybridCache::new(
                hybrid_cache_config,
                vb_m.dtype(),
                &normal_loading_metadata.real_device,
            )
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create hybrid cache: {e}")))?,
        ));

        let world_size = mapper.get_comm_for(0)?.world_size();
        let num_attention_heads = config.num_attention_heads / world_size;
        let num_kv_heads = (config.num_key_value_heads / world_size).max(1);

        Ok(Self {
            embed_tokens,
            layers,
            layer_types,
            norm,
            lm_head,
            local_cache,
            kv_cache: EitherCache::Hybrid(pipeline_cache),
            device: normal_loading_metadata.real_device,
            mapper,
            cfg: ModelConfigMetadata {
                max_seq_len: config.max_position_embeddings,
                num_layers: config.num_hidden_layers,
                hidden_size: config.hidden_size,
                num_kv_heads,
                num_attn_heads: num_attention_heads,
                sliding_window: config.sliding_window,
                k_head_dim: config.head_dim,
                v_head_dim: config.head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            num_attention_heads,
            max_seq_len: config.max_position_embeddings,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;

        let mut local_cache = self.local_cache.lock().unwrap_or_else(|e| e.into_inner());

        let mask = CausalMasker.make_causal_mask_matrix(
            input_ids,
            metadata
                .as_ref()
                .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                .unwrap_or(&*local_cache as &dyn PastKvLenCache),
            xs.dtype(),
            self.num_attention_heads,
        )?;
        let mask = mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });
        let mask = DeviceMappedMask::new(mask, &*self.mapper)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, layer_idx)?;

            match &layer.layer_impl {
                LayerImplVariant::FullAttention(_) => {
                    if let LocalLayerCache::Attention(kv_cache) =
                        &mut local_cache.caches[layer_idx]
                    {
                        if metadata
                            .as_ref()
                            .map(|(_, meta)| meta.is_first_prompt_chunk)
                            .unwrap_or(seqlen_offsets[0] == 0)
                        {
                            kv_cache.reset();
                        }
                        let mask_for_layer =
                            mask.as_ref().map(|m| m.get(xs.device()).clone());
                        xs = layer.forward_attention(
                            &xs,
                            &mask_for_layer,
                            seqlen_offsets,
                            kv_cache,
                            metadata.as_ref().map(|(kv_cache, meta)| {
                                (kv_cache[layer_idx].clone(), *meta)
                            }),
                            flash_params,
                        )?;
                    }
                }
                LayerImplVariant::LinearAttention(_) => {
                    if let LocalLayerCache::LinearAttention(gdn_cache) =
                        &mut local_cache.caches[layer_idx]
                    {
                        if metadata
                            .as_ref()
                            .map(|(_, meta)| meta.is_first_prompt_chunk)
                            .unwrap_or(seqlen_offsets[0] == 0)
                        {
                            gdn_cache.reset()?;
                        }
                        xs = layer.forward_linear(&xs, gdn_cache)?;
                    }
                }
            }
        }

        let xs = xs.to_device(&self.device)?;
        let xs = self.norm.forward(&xs)?;
        let mut xs = extract_logits(&xs, context_lens)?;
        if let Some(t) = self.lm_head.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        MatMul.qmethod_matmul(&xs, &*self.lm_head)
    }
}

// ====================== IsqModel ======================

impl IsqModel for Model {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            match &mut layer.layer_impl {
                LayerImplVariant::FullAttention(attn) => {
                    tensors.push((&mut attn.q_proj, Some(i)));
                    tensors.push((&mut attn.k_proj, Some(i)));
                    tensors.push((&mut attn.v_proj, Some(i)));
                    tensors.push((&mut attn.o_proj, Some(i)));
                }
                LayerImplVariant::LinearAttention(gdn) => {
                    // Include ALL GDN projections (input + output) for ISQ.
                    // Without this, the large in_proj_qkv / in_proj_z weights stay
                    // in full BF16, causing 4× higher memory bandwidth per decode
                    // step than the quantized attention layers.
                    for m in gdn.get_isq_layers() {
                        tensors.push((m, Some(i)));
                    }
                }
            }
            for m in layer.ffn.get_isq_layers() {
                tensors.push((m, Some(i)));
            }
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("norm").add(&self.norm);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);

            match &layer.layer_impl {
                LayerImplVariant::FullAttention(attn) => {
                    uvb_l.pp("self_attn").pp("q_norm").add(&attn.q_norm);
                    uvb_l.pp("self_attn").pp("k_norm").add(&attn.k_norm);
                }
                LayerImplVariant::LinearAttention(gdn) => {
                    let la = uvb_l.pp("linear_attn");
                    match &gdn.projection {
                        GdnProjection::SplitQkvZa {
                            in_proj_qkv,
                            in_proj_z,
                            in_proj_b,
                            in_proj_a,
                        } => {
                            la.pp("in_proj_qkv").add_tensor(
                                "weight",
                                in_proj_qkv.unquant_weight_bias().unwrap().0,
                            );
                            la.pp("in_proj_z").add_tensor(
                                "weight",
                                in_proj_z.unquant_weight_bias().unwrap().0,
                            );
                            la.pp("in_proj_b").add_tensor(
                                "weight",
                                in_proj_b.unquant_weight_bias().unwrap().0,
                            );
                            la.pp("in_proj_a").add_tensor(
                                "weight",
                                in_proj_a.unquant_weight_bias().unwrap().0,
                            );
                        }
                        GdnProjection::SplitQkvZaMerged {
                            in_proj_q,
                            in_proj_k,
                            in_proj_v,
                            in_proj_z,
                            in_proj_b,
                            in_proj_a,
                        } => {
                            la.pp("in_proj_q").add_tensor(
                                "weight",
                                in_proj_q.unquant_weight_bias().unwrap().0,
                            );
                            la.pp("in_proj_k").add_tensor(
                                "weight",
                                in_proj_k.unquant_weight_bias().unwrap().0,
                            );
                            la.pp("in_proj_v").add_tensor(
                                "weight",
                                in_proj_v.unquant_weight_bias().unwrap().0,
                            );
                            la.pp("in_proj_z").add_tensor(
                                "weight",
                                in_proj_z.unquant_weight_bias().unwrap().0,
                            );
                            la.pp("in_proj_b").add_tensor(
                                "weight",
                                in_proj_b.unquant_weight_bias().unwrap().0,
                            );
                            la.pp("in_proj_a").add_tensor(
                                "weight",
                                in_proj_a.unquant_weight_bias().unwrap().0,
                            );
                        }
                        GdnProjection::FusedAll {
                            in_proj,
                            qkv_end,
                            z_end,
                            b_end,
                        } => {
                            // The fused weight covers [qkv | z | b | a].
                            // Split it back into the four named checkpoint
                            // slices so that save/reload works correctly.
                            if let Some((w, _)) = in_proj.unquant_weight_bias() {
                                let total = w.dim(0).unwrap_or(0);
                                let a_end = total;
                                if let Ok(w_qkv) = w.narrow(0, 0, *qkv_end) {
                                    la.pp("in_proj_qkv").add_tensor("weight", w_qkv);
                                }
                                if let Ok(w_z) = w.narrow(0, *qkv_end, *z_end - *qkv_end) {
                                    la.pp("in_proj_z").add_tensor("weight", w_z);
                                }
                                if let Ok(w_b) = w.narrow(0, *z_end, *b_end - *z_end) {
                                    la.pp("in_proj_b").add_tensor("weight", w_b);
                                }
                                if let Ok(w_a) = w.narrow(0, *b_end, a_end - *b_end) {
                                    la.pp("in_proj_a").add_tensor("weight", w_a);
                                }
                            }
                        }
                        GdnProjection::FusedQkvzBa { .. } => {
                            // Not used for Qwen3.5 text model
                        }
                    }
                    la.add_tensor("conv1d.weight", gdn.conv1d_weight.clone());
                    la.add_tensor("dt_bias", gdn.dt_bias.clone());
                    la.add_tensor("A_log", gdn.a_log.clone());
                    la.pp("norm").add_tensor("weight", gdn.norm.weight.clone());
                }
            }
        }

        uvb.to_safetensors()
    }
}

// ====================== NormalModel ======================

impl NormalModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.forward(input_ids, seqlen_offsets, context_lens, metadata, flash_params)
    }

    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        unimplemented!("Qwen3.5 does not support X-LoRA")
    }

    fn cache(&self) -> &EitherCache {
        &self.kv_cache
    }

    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.kv_cache
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn is_xlora(&self) -> bool {
        false
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl AnyMoeBaseModelMixin for Model {}
