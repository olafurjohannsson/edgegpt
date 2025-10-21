//! GPT-specific configuration

use serde::{Deserialize, Serialize};
use edgetransformers::config::TransformerConfig;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GPTConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    pub n_ctx: usize,  // max sequence length
    pub n_embd: usize,  // hidden size
    pub n_layer: usize,  // number of layers
    pub n_head: usize,  // number of attention heads
    pub layer_norm_epsilon: f32,
    pub initializer_range: f32,
    #[serde(default = "default_activation")]
    pub activation_function: String,
    #[serde(default = "default_dropout")]
    pub resid_pdrop: f32,
    #[serde(default = "default_dropout")]
    pub embd_pdrop: f32,
    #[serde(default = "default_dropout")]
    pub attn_pdrop: f32,
    pub model_type: String,
}

fn default_activation() -> String {
    "gelu".to_string()
}

fn default_dropout() -> f32 {
    0.1
}

fn default_vocab_size() -> usize {
    50257
}

impl TransformerConfig for GPTConfig {
    fn hidden_size(&self) -> usize { self.n_embd }
    fn num_attention_heads(&self) -> usize { self.n_head }
    fn num_hidden_layers(&self) -> usize { self.n_layer }
    fn max_position_embeddings(&self) -> usize { self.n_ctx }
    fn vocab_size(&self) -> usize { self.vocab_size }
    fn intermediate_size(&self) -> usize { self.n_embd * 4 }  // GPT typically uses 4x hidden for FFN
    fn layer_norm_eps(&self) -> f32 { self.layer_norm_epsilon }
    fn hidden_dropout_prob(&self) -> f32 { self.resid_pdrop }
    fn attention_dropout_prob(&self) -> f32 { self.attn_pdrop }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BartConfig {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub decoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_ffn_dim: usize,
    pub d_model: usize, // This is the hidden_size for BART
    pub activation_function: String,
    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f32,
    #[serde(default = "default_dropout")]
    pub dropout: f32,
    // Token IDs
    pub pad_token_id: u32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    #[serde(default)]
    pub decoder_start_token_id: u32,
    pub scale_embedding: bool,
}

fn default_layer_norm_epsilon() -> f32 { 1e-5 }

// Implement the trait so it can be used generically if needed
impl TransformerConfig for BartConfig {
    fn hidden_size(&self) -> usize { self.d_model }
    fn num_attention_heads(&self) -> usize { self.encoder_attention_heads } // Or decoder, they are often the same
    fn num_hidden_layers(&self) -> usize { self.encoder_layers }
    fn max_position_embeddings(&self) -> usize { self.max_position_embeddings }
    fn vocab_size(&self) -> usize { self.vocab_size }
    fn intermediate_size(&self) -> usize { self.encoder_ffn_dim }
    fn layer_norm_eps(&self) -> f32 { self.layer_norm_epsilon }
    fn hidden_dropout_prob(&self) -> f32 { self.dropout }
    fn attention_dropout_prob(&self) -> f32 { self.dropout }
}