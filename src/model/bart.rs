//! BART Encoder-Decoder model implementations

use crate::config::BartConfig;
use crate::ModelWeights;
use anyhow::Result;
use edgetransformers::{Embeddings, FeedForward, LayerNorm, MultiHeadAttention, TransformerConfig};
use ndarray::{s, Array2, Array3, Axis};

#[cfg(not(target_arch = "wasm32"))]
use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
use crate::tokenizer::wasm::BPETokenizer as Tokenizer;

/// BART layer

// A standard Transformer layer for the encoder.
pub struct BartEncoderLayer {
    self_attn: MultiHeadAttention,
    self_attn_layer_norm: LayerNorm,
    ffn: FeedForward,
    ffn_layer_norm: LayerNorm,
}
impl BartEncoderLayer {
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
    ) -> Result<Array3<f32>> {
        // --- Self-Attention Block (Post-Norm) ---
        let residual = hidden_states;
        let attn_output = self
            .self_attn
            .forward_bart(hidden_states, None, Some(attention_mask), false)?;
        let mut hidden_states = residual + &attn_output;
        hidden_states = self.self_attn_layer_norm.forward_3d(&hidden_states);

        // --- Feed-Forward Block (Post-Norm) ---
        let residual = &hidden_states;
        let ffn_output = self.ffn.forward(&hidden_states)?;
        hidden_states = residual + &ffn_output;
        hidden_states = self.ffn_layer_norm.forward_3d(&hidden_states);

        Ok(hidden_states)
    }
}
fn create_encoder_attention_mask(attention_mask: &Array2<f32>) -> Array2<f32> {
    // BART encoder mask: 1 = attend, 0 = don't attend (padding)
    // Your shared function expects this format, so just return as-is
    attention_mask.clone()
}

/// Create BART decoder causal mask (lower triangular)
fn create_decoder_causal_mask(seq_len: usize) -> Array2<f32> {
    let mut mask = Array2::<f32>::zeros((seq_len, seq_len));
    
    // Lower triangular: 1 = can attend, 0 = cannot attend
    for i in 0..seq_len {
        for j in 0..=i {
            mask[[i, j]] = 1.0;  // Can attend to past and current
        }
    }
    // Upper triangle stays 0 (cannot attend to future)
    
    mask
}
pub struct BartEncoder {
    layers: Vec<BartEncoderLayer>,
}

impl BartEncoder {
    pub fn forward(
        &self,
        mut hidden_states: Array3<f32>,
        attention_mask: &Array2<f32>,
    ) -> Result<Array3<f32>> {
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }
        Ok(hidden_states)
    }
}

pub struct BartDecoderLayer {
    // Two attention blocks for encoder-decoder
    self_attn: MultiHeadAttention,
    self_attn_layer_norm: LayerNorm,
    cross_attn: MultiHeadAttention,
    cross_attn_layer_norm: LayerNorm,
    ffn: FeedForward,
    ffn_layer_norm: LayerNorm,
}

impl BartDecoderLayer {
        pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        decoder_causal_mask: Option<&Array2<f32>>,
        encoder_attention_mask: &Array2<f32>,
    ) -> Result<Array3<f32>> {
        
        // Self-Attention
        let residual = hidden_states;
        let self_attn_output = self.self_attn.forward_bart(
            hidden_states, 
            None, 
            decoder_causal_mask,
            true
        )?;
        
        let mut hidden_states = residual + &self_attn_output;
        
        hidden_states = self.self_attn_layer_norm.forward_3d(&hidden_states);

        // Cross-Attention
        let residual = &hidden_states;
        let cross_attn_output = self.cross_attn.forward_bart(
            &hidden_states,
            Some(encoder_hidden_states),
            Some(encoder_attention_mask),
            false
        )?;
        
        hidden_states = residual + &cross_attn_output;
        
        hidden_states = self.cross_attn_layer_norm.forward_3d(&hidden_states);
        // FFN
        let residual = &hidden_states;
        let ffn_output = self.ffn.forward(&hidden_states)?;
        
        hidden_states = residual + &ffn_output;
        hidden_states = self.ffn_layer_norm.forward_3d(&hidden_states);

        Ok(hidden_states)
    }
}

pub struct BartDecoder {
    layers: Vec<BartDecoderLayer>,
}

impl BartDecoder {
    pub fn forward(
        &self,
        mut hidden_states: Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        decoder_causal_mask: Option<&Array2<f32>>,
        encoder_attention_mask: &Array2<f32>,
    ) -> Result<Array3<f32>> {
        for layer in &self.layers {
            hidden_states = layer.forward(
                &hidden_states,
                encoder_hidden_states,
                decoder_causal_mask,
                encoder_attention_mask,
            )?;
        }
        Ok(hidden_states)
    }
}

// The main model
pub struct BartModel {
    pub shared_embeddings: Array2<f32>, //  word embedding matrix
    pub encoder_pos_embeddings: Array2<f32>,
    pub decoder_pos_embeddings: Array2<f32>,
    pub encoder_embed_layer_norm: LayerNorm,
    pub decoder_embed_layer_norm: LayerNorm,
    pub encoder: BartEncoder,
    pub decoder: BartDecoder,
    pub lm_head: Array2<f32>, // projection to vocabulary
    pub config: BartConfig,
    pub tokenizer: Tokenizer,
}

impl BartModel {
    pub fn from_weights(
        weights: &ModelWeights,
        config: BartConfig,
        tokenizer: Tokenizer,
    ) -> Result<Self> {
        let shared_embeddings = weights.get_array2("model.shared.weight")?;
        // diffrnent embeddings for encoder and decoder
        let encoder_pos_embeddings = weights.get_array2("model.encoder.embed_positions.weight")?;
        let decoder_pos_embeddings = weights.get_array2("model.decoder.embed_positions.weight")?;

        let encoder_embed_layer_norm = LayerNorm::new(
            weights.get_array1("model.encoder.layernorm_embedding.weight")?,
            weights.get_array1("model.encoder.layernorm_embedding.bias")?,
            config.layer_norm_epsilon,
        );
        let decoder_embed_layer_norm = LayerNorm::new(
            weights.get_array1("model.decoder.layernorm_embedding.weight")?,
            weights.get_array1("model.decoder.layernorm_embedding.bias")?,
            config.layer_norm_epsilon,
        );

        // let word_embeddings = weights.get_array2("model.shared.weight")?;
        // let position_embeddings = weights.get_array2("model.encoder.embed_positions.weight")?;

        // let token_type_embeddings = Array2::zeros((config.max_position_embeddings, config.d_model));

        let lm_head = shared_embeddings.clone();
        // let embeddings =
        //     Embeddings::new(word_embeddings, position_embeddings, token_type_embeddings);

        let mut encoder_layers = Vec::new();
        for i in 0..config.encoder_layers {
            let prefix = format!("model.encoder.layers.{}", i);
            let attn = MultiHeadAttention::new(
                config.d_model,
                config.encoder_attention_heads,
                weights
                    .get_array2(&format!("{}.self_attn.q_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.self_attn.q_proj.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.self_attn.k_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.self_attn.k_proj.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.self_attn.v_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.self_attn.v_proj.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.self_attn.out_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.self_attn.out_proj.bias", prefix))?,
            );
            let self_attn_layer_norm = LayerNorm::new(
                weights.get_array1(&format!("{}.self_attn_layer_norm.weight", prefix))?,
                weights.get_array1(&format!("{}.self_attn_layer_norm.bias", prefix))?,
                config.layer_norm_epsilon,
            );
            let fc1_weight = weights.get_array2(&format!("{}.fc1.weight", prefix))?;
            let fc2_weight = weights.get_array2(&format!("{}.fc2.weight", prefix))?;

            let ffn = FeedForward::new(
                // transpose fc weights because edgetransformers does not
                fc1_weight.t().to_owned(),
                weights.get_array1(&format!("{}.fc1.bias", prefix))?,
                fc2_weight.t().to_owned(),
                weights.get_array1(&format!("{}.fc2.bias", prefix))?,
            );
            let ffn_layer_norm = LayerNorm::new(
                weights.get_array1(&format!("{}.final_layer_norm.weight", prefix))?,
                weights.get_array1(&format!("{}.final_layer_norm.bias", prefix))?,
                config.layer_norm_epsilon,
            );
            encoder_layers.push(BartEncoderLayer {
                self_attn: attn,
                self_attn_layer_norm,
                ffn,
                ffn_layer_norm,
            });
        }
        let encoder = BartEncoder {
            layers: encoder_layers,
        };

        let mut decoder_layers = Vec::new();
        for i in 0..config.decoder_layers {
            let prefix = format!("model.decoder.layers.{}", i);
            let self_attn = MultiHeadAttention::new(
                config.d_model,
                config.decoder_attention_heads,
                weights
                    .get_array2(&format!("{}.self_attn.q_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.self_attn.q_proj.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.self_attn.k_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.self_attn.k_proj.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.self_attn.v_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.self_attn.v_proj.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.self_attn.out_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.self_attn.out_proj.bias", prefix))?,
            );
            let self_attn_layer_norm = LayerNorm::new(
                weights.get_array1(&format!("{}.self_attn_layer_norm.weight", prefix))?,
                weights.get_array1(&format!("{}.self_attn_layer_norm.bias", prefix))?,
                config.layer_norm_epsilon,
            );
            // cross-attention block
            let cross_attn = MultiHeadAttention::new(
                config.d_model,
                config.decoder_attention_heads,
                weights
                    .get_array2(&format!("{}.encoder_attn.q_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.encoder_attn.q_proj.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.encoder_attn.k_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.encoder_attn.k_proj.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.encoder_attn.v_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.encoder_attn.v_proj.bias", prefix))?,
                weights
                    .get_array2(&format!("{}.encoder_attn.out_proj.weight", prefix))?
                    .t()
                    .to_owned(),
                weights.get_array1(&format!("{}.encoder_attn.out_proj.bias", prefix))?,
            );
            let cross_attn_layer_norm = LayerNorm::new(
                weights.get_array1(&format!("{}.encoder_attn_layer_norm.weight", prefix))?,
                weights.get_array1(&format!("{}.encoder_attn_layer_norm.bias", prefix))?,
                config.layer_norm_epsilon,
            );
            let fc1_weight_dec = weights.get_array2(&format!("{}.fc1.weight", prefix))?;
            let fc2_weight_dec = weights.get_array2(&format!("{}.fc2.weight", prefix))?;

            let ffn = FeedForward::new(
                // transpose before going into feedforward
                fc1_weight_dec.t().to_owned(),
                weights.get_array1(&format!("{}.fc1.bias", prefix))?,
                fc2_weight_dec.t().to_owned(),
                weights.get_array1(&format!("{}.fc2.bias", prefix))?,
            );
            let ffn_layer_norm = LayerNorm::new(
                weights.get_array1(&format!("{}.final_layer_norm.weight", prefix))?,
                weights.get_array1(&format!("{}.final_layer_norm.bias", prefix))?,
                config.layer_norm_epsilon,
            );
            decoder_layers.push(BartDecoderLayer {
                self_attn,
                self_attn_layer_norm,
                cross_attn,
                cross_attn_layer_norm,
                ffn,
                ffn_layer_norm,
            });
        }
        let decoder = BartDecoder {
            layers: decoder_layers,
        };

        Ok(Self {
            shared_embeddings,
            encoder_pos_embeddings,
            decoder_pos_embeddings,
            encoder_embed_layer_norm,
            decoder_embed_layer_norm,
            encoder,
            decoder,
            lm_head,
            config,
            tokenizer,
        })
    }

    pub fn embed(&self, input_ids: &Array2<f32>, is_decoder: bool) -> Array3<f32> {
        let (batch_size, seq_len) = input_ids.dim();
        // todo: 
        let embed_scale = if self.config.scale_embedding {
            (self.config.d_model as f32).sqrt()
        } else {
            1.0
        };
        let mut hidden = Array3::<f32>::zeros((batch_size, seq_len, self.config.d_model));
        for i in 0..batch_size {
            for j in 0..seq_len {
                let token_id = input_ids[[i, j]] as usize;
                hidden
                    .slice_mut(s![i, j, ..])
                    .assign(&self.shared_embeddings.row(token_id));
            }
        }

        // --- FIX #2 (continued): Use the correct positional embeddings and LayerNorm ---
        let (pos_embeddings, layer_norm) = if is_decoder {
            (&self.decoder_pos_embeddings, &self.decoder_embed_layer_norm)
        } else {
            (&self.encoder_pos_embeddings, &self.encoder_embed_layer_norm)
        };

        let pos_embeddings_slice = pos_embeddings
            .slice(s![2..seq_len + 2, ..])
            .insert_axis(Axis(0));
        hidden += &pos_embeddings_slice;

        // Apply the correct LayerNorm depending on whether we're in the encoder or decoder
        layer_norm.forward_3d(&hidden)
    }
}
