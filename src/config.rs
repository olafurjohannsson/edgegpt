//! GPT-specific configuration

use serde::{Deserialize, Serialize};
use edgetransformers::config::TransformerConfig;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GPTConfig {
    pub n_vocab: usize,
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

impl TransformerConfig for GPTConfig {
    fn hidden_size(&self) -> usize { self.n_embd }
    fn num_attention_heads(&self) -> usize { self.n_head }
    fn num_hidden_layers(&self) -> usize { self.n_layer }
    fn max_position_embeddings(&self) -> usize { self.n_ctx }
    fn vocab_size(&self) -> usize { self.n_vocab }
    fn intermediate_size(&self) -> usize { self.n_embd * 4 }  // GPT typically uses 4x hidden for FFN
    fn layer_norm_eps(&self) -> f32 { self.layer_norm_epsilon }
    fn hidden_dropout_prob(&self) -> f32 { self.resid_pdrop }
    fn attention_dropout_prob(&self) -> f32 { self.attn_pdrop }
}