//! Text generation utilities

use anyhow::Result;
use ndarray::{Array1, Array2, Array3, Array4, Axis};
use rand::Rng;
use crate::model::base::GPTBase;

#[cfg(not(target_arch = "wasm32"))]
use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
use crate::tokenizer::wasm::BPETokenizer as Tokenizer;

/// Configuration for text generation
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    pub sampling_strategy: SamplingStrategy,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 50,
            temperature: 1.0,
            top_k: Some(50),
            top_p: Some(0.95),
            repetition_penalty: 1.0,
            sampling_strategy: SamplingStrategy::TopKTopP,
            eos_token_id: None,
            pad_token_id: None,
        }
    }
}

/// Sampling strategies for generation
#[derive(Clone, Debug)]
pub enum SamplingStrategy {
    Greedy,
    TopK,
    TopP,
    TopKTopP,
    Temperature,
}

/// Generate text autoregressively
pub fn generate_text(
    model: &GPTBase,
    tokenizer: &Tokenizer,
    prompt: &str,
    config: &GenerationConfig,
) -> Result<String> {
    // Tokenize prompt
    #[cfg(not(target_arch = "wasm32"))]
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    
    #[cfg(target_arch = "wasm32")]
    let encoding = tokenizer.encode(prompt, 512)?;
    
    let mut input_ids = encoding.get_ids().to_vec();
    let batch_size = 1;
    
    // Initialize past key-values
    let mut past: Option<Vec<(Array4<f32>, Array4<f32>)>> = None;
    
    // Generation loop
    for _ in 0..config.max_new_tokens {
        // Prepare current input
        let cur_len = if past.is_some() { 1 } else { input_ids.len() };
        let start_idx = if past.is_some() { input_ids.len() - 1 } else { 0 };
        
        let mut input_array = Array2::<f32>::zeros((batch_size, cur_len));
        for (j, &id) in input_ids[start_idx..].iter().enumerate() {
            input_array[[0, j]] = id as f32;
        }
        
        // Forward pass
        let (hidden_states, presents) = model.forward(&input_array, past)?;
        past = Some(presents);
        
        // Get logits for last position
        let logits = model.get_logits(&hidden_states);
        let next_token_logits = logits.slice(ndarray::s![0, -1, ..]).to_owned();
        
        // Apply repetition penalty
        let next_token_logits = apply_repetition_penalty(
            next_token_logits,
            &input_ids,
            config.repetition_penalty,
        );
        
        // Sample next token
        let next_token = sample_token(
            next_token_logits,
            config,
        )?;
        
        // Check for EOS
        if let Some(eos_id) = config.eos_token_id {
            if next_token == eos_id {
                break;
            }
        }
        
        input_ids.push(next_token);
    }
    
    // Decode back to text
    #[cfg(not(target_arch = "wasm32"))]
    let output = tokenizer
        .decode(&input_ids, true)
        .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;
    
    #[cfg(target_arch = "wasm32")]
    let output = tokenizer.decode(&input_ids)?;
    
    Ok(output)
}

/// Apply repetition penalty to logits
fn apply_repetition_penalty(
    mut logits: Array1<f32>,
    generated_ids: &[u32],
    penalty: f32,
) -> Array1<f32> {
    if penalty == 1.0 {
        return logits;
    }
    
    for &id in generated_ids {
        let idx = id as usize;
        if logits[idx] < 0.0 {
            logits[idx] *= penalty;
        } else {
            logits[idx] /= penalty;
        }
    }
    
    logits
}

/// Sample a token from logits
fn sample_token(
    mut logits: Array1<f32>,
    config: &GenerationConfig,
) -> Result<u32> {
    let mut rng = rand::thread_rng();
    
    match config.sampling_strategy {
        SamplingStrategy::Greedy => {
            // Argmax
            Ok(logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap())
        }
        
        SamplingStrategy::Temperature => {
            // Apply temperature
            logits /= config.temperature;
            
            // Softmax and sample
            let probs = softmax_1d(&logits);
            sample_from_probs(&probs, &mut rng)
        }
        
        SamplingStrategy::TopK => {
            // Apply top-k filtering
            if let Some(k) = config.top_k {
                logits = top_k_filtering(logits, k);
            }
            
            // Apply temperature
            logits /= config.temperature;
            
            // Softmax and sample
            let probs = softmax_1d(&logits);
            sample_from_probs(&probs, &mut rng)
        }
        
        SamplingStrategy::TopP => {
            // Apply top-p (nucleus) filtering
            if let Some(p) = config.top_p {
                logits = top_p_filtering(logits, p);
            }
            
            // Apply temperature
            logits /= config.temperature;
            
            // Softmax and sample
            let probs = softmax_1d(&logits);
            sample_from_probs(&probs, &mut rng)
        }
        
        SamplingStrategy::TopKTopP => {
            // Apply both top-k and top-p
            if let Some(k) = config.top_k {
                logits = top_k_filtering(logits, k);
            }
            if let Some(p) = config.top_p {
                logits = top_p_filtering(logits, p);
            }
            
            // Apply temperature
            logits /= config.temperature;
            
            // Softmax and sample
            let probs = softmax_1d(&logits);
            sample_from_probs(&probs, &mut rng)
        }
    }
}

/// Apply softmax to 1D array
fn softmax_1d(logits: &Array1<f32>) -> Array1<f32> {
    let max_val = logits.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let exp_logits = (logits - max_val).mapv(f32::exp);
    let sum_exp = exp_logits.sum();
    exp_logits / sum_exp
}

/// Top-k filtering
fn top_k_filtering(mut logits: Array1<f32>, k: usize) -> Array1<f32> {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    
    // Set all but top-k to -inf
    for &idx in &indices[k..] {
        logits[idx] = f32::NEG_INFINITY;
    }
    
    logits
}

/// Top-p (nucleus) filtering
fn top_p_filtering(mut logits: Array1<f32>, p: f32) -> Array1<f32> {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    
    let probs = softmax_1d(&logits);
    let mut cumulative = 0.0;
    let mut cutoff_idx = 0;
    
    for (i, &idx) in indices.iter().enumerate() {
        cumulative += probs[idx];
        if cumulative > p {
            cutoff_idx = i + 1;
            break;
        }
    }
    
    // Set all but nucleus to -inf
    for &idx in &indices[cutoff_idx..] {
        logits[idx] = f32::NEG_INFINITY;
    }
    
    logits
}

/// Sample from probability distribution
fn sample_from_probs(probs: &Array1<f32>, rng: &mut impl Rng) -> Result<u32> {
    let uniform: f32 = rng.gen();
    let mut cumulative = 0.0;
    
    for (idx, &prob) in probs.iter().enumerate() {
        cumulative += prob;
        if cumulative >= uniform {
            return Ok(idx as u32);
        }
    }
    
    // Fallback to last index
    Ok((probs.len() - 1) as u32)
}