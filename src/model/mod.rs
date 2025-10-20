//! GPT model implementations

pub mod base;
pub mod distilgpt2;
pub mod gpt2;

use anyhow::Result;
use std::path::PathBuf;

pub use base::GPTBase;
pub use distilgpt2::DistilGPT2;
pub use gpt2::GPT2;

#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    DistilGPT2,
    GPT2,
    GPT2Medium,
}

/// Main GPT model for text generation
pub enum GPTModel {
    DistilGPT2(DistilGPT2),
    GPT2(GPT2),
}

impl GPTModel {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_pretrained(model_type: ModelType) -> Result<Self> {
        use crate::weights::ModelWeights;
        use crate::config::GPTConfig;
        
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("edgegpt");
        
        std::fs::create_dir_all(&cache_dir)?;
        
        let model_path = match model_type {
            ModelType::DistilGPT2 => cache_dir.join("distilgpt2"),
            ModelType::GPT2 => cache_dir.join("gpt2"),
            ModelType::GPT2Medium => cache_dir.join("gpt2-medium"),
        };
        
        ensure_model_files(model_type, &model_path)?;
        
        let weights = ModelWeights::load(&model_path)?;
        let config: GPTConfig = serde_json::from_str(&weights.config_json)?;
        
        let tokenizer_file = model_path.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_file)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        
        match model_type {
            ModelType::DistilGPT2 => {
                Ok(GPTModel::DistilGPT2(
                    DistilGPT2::from_weights(weights, tokenizer, config)?
                ))
            }
            ModelType::GPT2 | ModelType::GPT2Medium => {
                Ok(GPTModel::GPT2(
                    GPT2::from_weights(weights, tokenizer, config)?
                ))
            }
        }
    }
    
    pub fn generate(
        &self,
        prompt: &str,
        config: &crate::generation::GenerationConfig,
    ) -> Result<String> {
        match self {
            GPTModel::DistilGPT2(model) => model.generate(prompt, config),
            GPTModel::GPT2(model) => model.generate(prompt, config),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure_model_files(model_type: ModelType, model_path: &std::path::Path) -> Result<()> {
    let (weights_url, tokenizer_url, config_url) = match model_type {
        ModelType::DistilGPT2 => (
            "https://huggingface.co/distilbert/distilgpt2/resolve/main/model.safetensors",
            "https://huggingface.co/distilbert/distilgpt2/resolve/main/tokenizer.json",
            "https://huggingface.co/distilbert/distilgpt2/resolve/main/config.json",
        ),
        ModelType::GPT2 => (
            "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors",
            "https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json",
            "https://huggingface.co/openai-community/gpt2/resolve/main/config.json",
        ),
        ModelType::GPT2Medium => (
            "https://huggingface.co/openai-community/gpt2-medium/resolve/main/model.safetensors",
            "https://huggingface.co/openai-community/gpt2-medium/resolve/main/tokenizer.json",
            "https://huggingface.co/openai-community/gpt2-medium/resolve/main/config.json",
        ),
    };
    
    std::fs::create_dir_all(model_path)?;
    
    let download = |url: &str, path: &std::path::Path| -> anyhow::Result<()> {
        if !path.exists() {
            let resp = reqwest::blocking::get(url)?;
            if !resp.status().is_success() {
                anyhow::bail!("Failed to download {}: {}", url, resp.status());
            }
            let bytes = resp.bytes()?;
            std::fs::write(path, &bytes)?;
        }
        Ok(())
    };
    
    download(weights_url, &model_path.join("model.safetensors"))?;
    download(tokenizer_url, &model_path.join("tokenizer.json"))?;
    download(config_url, &model_path.join("config.json"))?;
    
    Ok(())
}