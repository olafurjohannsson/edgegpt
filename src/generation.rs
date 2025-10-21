//! Text generation utilities

use crate::model::bart::BartModel;
use crate::model::base::GPTBase;
use anyhow::Result;
use edgetransformers::TransformerConfig;
use ndarray::{s, Array1, Array2, Array3, Array4, Axis};
use rand::Rng;

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
            temperature: 0.7,
            top_k: Some(50),
            top_p: Some(0.9),
            repetition_penalty: 1.1,
            sampling_strategy: SamplingStrategy::TopKTopP,
            eos_token_id: Some(50256), // default GPT-2 EOS
            pad_token_id: Some(50256),
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

/// Streaming token callback
pub type TokenCallback = Box<dyn FnMut(u32, &str) -> bool>;

/// Generate text with streaming support
pub fn generate_text_streaming(
    model: &GPTBase,
    tokenizer: &Tokenizer,
    prompt: &str,
    config: &GenerationConfig,
    mut on_token: TokenCallback,
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
    let vocab_size = model.config.vocab_size();

    // Initialize past key-values
    let mut past: Option<Vec<(Array4<f32>, Array4<f32>)>> = None;

    // Generation loop
    for _ in 0..config.max_new_tokens {
        // Prepare current input
        let cur_len = if past.is_some() { 1 } else { input_ids.len() };
        let start_idx = if past.is_some() {
            input_ids.len() - 1
        } else {
            0
        };

        let mut input_array = Array2::<f32>::zeros((batch_size, cur_len));
        for (j, &id) in input_ids[start_idx..].iter().enumerate() {
            input_array[[0, j]] = id as f32;
        }

        // Forward pass
        let (hidden_states, presents) = model.forward(&input_array, past)?;
        past = Some(presents);

        // Get logits for last position
        let logits = model.get_logits(&hidden_states);
        let next_token_logits = logits.slice(s![0, -1, ..]).to_owned();

        // Ensure we're within vocab bounds
        let mut bounded_logits = Array1::<f32>::from_elem(vocab_size, f32::NEG_INFINITY);
        let actual_size = next_token_logits.len().min(vocab_size);
        bounded_logits
            .slice_mut(s![..actual_size])
            .assign(&next_token_logits.slice(s![..actual_size]));

        // Apply repetition penalty
        let bounded_logits =
            apply_repetition_penalty(bounded_logits, &input_ids, config.repetition_penalty);

        // Sample next token
        let next_token = sample_token(bounded_logits, config)?.min(vocab_size as u32 - 1); // Ensure within vocab

        // Decode the new token
        #[cfg(not(target_arch = "wasm32"))]
        let token_text = tokenizer.decode(&[next_token], false).unwrap_or_default();

        #[cfg(target_arch = "wasm32")]
        let token_text = tokenizer.decode(&[next_token]).unwrap_or_default();

        // Call the streaming callback (returns false to stop)
        if !on_token(next_token, &token_text) {
            break;
        }

        // Check for EOS
        if let Some(eos_id) = config.eos_token_id {
            if next_token == eos_id {
                break;
            }
        }

        input_ids.push(next_token);
    }

    // Decode complete output
    #[cfg(not(target_arch = "wasm32"))]
    let output = tokenizer
        .decode(&input_ids, true)
        .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

    #[cfg(target_arch = "wasm32")]
    let output = tokenizer.decode(&input_ids)?;

    Ok(output)
}

/// Generate text autoregressively
pub fn generate_text(
    model: &GPTBase,
    tokenizer: &Tokenizer,
    prompt: &str,
    config: &GenerationConfig,
) -> Result<String> {
    // Use streaming internally but ignore callbacks
    generate_text_streaming(
        model,
        tokenizer,
        prompt,
        config,
        Box::new(|_, _| true), // Always continue
    )
}
type LayerCache = (Array4<f32>, Array4<f32>);
// The full cache for the decoder is a Vec of these self-attention caches.
type FullCache = Vec<LayerCache>;
pub fn generate_encoder_decoder(
    model: &BartModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    config: &GenerationConfig,
) -> Result<String> {
    println!("generate_encoder_decoder");
    let input_encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

    let min_length = 56; // TODO: from config

    // --- 1. Encoder Setup (This part is correct) ---
    let mut input_ids_vec = Vec::with_capacity(input_encoding.len() + 2);
    input_ids_vec.push(model.config.bos_token_id);
    input_ids_vec.extend(input_encoding.get_ids());
    input_ids_vec.push(model.config.eos_token_id);
    let attention_mask_vec = vec![1; input_ids_vec.len()];
    let mut input_ids_array = Array2::<f32>::zeros((1, input_ids_vec.len()));
    let mut encoder_mask_array = Array2::<f32>::zeros((1, input_ids_vec.len()));
    for (j, &id) in input_ids_vec.iter().enumerate() {
        input_ids_array[[0, j]] = id as f32;
    }
    for (j, &mask_val) in attention_mask_vec.iter().enumerate() {
        encoder_mask_array[[0, j]] = mask_val as f32;
    }

    // --- 2. Encode Once (This part is correct) ---
    let encoder_embeddings = model.embed(&input_ids_array, false, 0);
    let encoder_hidden_states = model
        .encoder
        .forward(encoder_embeddings, &encoder_mask_array)?;

    // --- 3. Decoder and Loop Setup ---
    let mut decoder_input_ids = vec![model.config.bos_token_id];
    let mut generated_tokens = vec![];
    let mut past_cache: Option<FullCache> = None;

    // --- 4. Generation Loop ---
    for step in 0..config.max_new_tokens {
        let last_token_id = *decoder_input_ids.last().unwrap();
        let mut decoder_input_array = Array2::<f32>::zeros((1, 1));
        decoder_input_array[[0, 0]] = last_token_id as f32;

        let past_len = step;
        let decoder_embeddings = model.embed(&decoder_input_array, true, past_len);

        // --- THE FINAL FIX IS HERE ---
        // 1. Re-introduce the causal mask creation INSIDE the loop.
        // Its size must be for the TOTAL sequence length so far (step + 1).
        let causal_mask = create_decoder_causal_mask(step + 1);

        // 2. Pass the newly created causal mask to the decoder.
        let (decoder_output, present_cache) = model.decoder.forward(
            decoder_embeddings,
            &encoder_hidden_states,
            Some(&causal_mask), // <-- Pass the mask here
            &encoder_mask_array,
            past_cache.as_ref(),
        )?;

        past_cache = Some(present_cache);

        // --- The rest of the loop remains the same ---
        let last_token_hidden_state = decoder_output.slice(s![0, -1, ..]);
        let mut logits: Array1<f32> = model.lm_head.dot(&last_token_hidden_state);

        if generated_tokens.len() < min_length {
            logits[model.config.eos_token_id as usize] = f32::NEG_INFINITY;
        }
        
        println!("\n=== STEP {} ===", step);
        logits = apply_repetition_penalty(logits, &decoder_input_ids, config.repetition_penalty);

        let mut top_logits: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        top_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("Top 5 logits: {:?}", &top_logits[..5]);

        let next_token_id = sample_token(logits, config)?;
        println!("Sampled token ID: {}", next_token_id);
        
        let token_text = tokenizer.decode(&[next_token_id], false).unwrap_or_default();
        println!("Token text: '{}'", token_text);

        if next_token_id == model.config.eos_token_id && generated_tokens.len() >= min_length {
            println!("HIT EOS - STOPPING (after min_length)");
            break;
        }

        generated_tokens.push(next_token_id);
        decoder_input_ids.push(next_token_id);
    }

    // --- 5. Decode and Return ---
    println!("\nGenerated {} tokens total", generated_tokens.len());
    println!("Token IDs: {:?}", generated_tokens);

    tokenizer
        .decode(&generated_tokens, true)
        .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))
}
fn create_decoder_causal_mask(seq_len: usize) -> Array2<f32> {
    let mut mask = Array2::<f32>::zeros((seq_len, seq_len));

    // Lower triangular: 1 = can attend, 0 = cannot attend
    for i in 0..seq_len {
        for j in 0..=i {
            mask[[i, j]] = 1.0; // Can attend to past and current
        }
    }
    // Upper triangle stays 0 (cannot attend to future)

    mask
}
// pub fn generate_encoder_decoder(
//     model: &BartModel,
//     tokenizer: &Tokenizer,
//     prompt: &str,
//     config: &GenerationConfig,
// ) -> Result<String> {
//     let input_encoding = tokenizer
//         .encode(prompt, false)
//         .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

//     // Manually add BOS and EOS tokens to the input sequence
//     let mut input_ids_vec = Vec::with_capacity(input_encoding.len() + 2);
//     input_ids_vec.push(model.config.bos_token_id); // Add BOS at the start
//     input_ids_vec.extend(input_encoding.get_ids());
//     input_ids_vec.push(model.config.eos_token_id); // Add EOS at the end

//     let attention_mask_vec = vec![1; input_ids_vec.len()];

//     let batch_size = 1;
//     let seq_len = input_ids_vec.len();

//     let mut input_ids_array = Array2::<f32>::zeros((batch_size, seq_len));
//     let mut encoder_mask_array = Array2::<f32>::zeros((batch_size, seq_len));

//     for (j, (&id, &mask_val)) in input_ids_vec.iter().zip(attention_mask_vec.iter()).enumerate() {
//         input_ids_array[[0, j]] = id as f32;
//         encoder_mask_array[[0, j]] = mask_val as f32;
//     }

//     // --- 1. INPUT ---
//     println!("--- 1. RUST INPUT ---");
//     println!("Input IDs: {:?}", input_ids_vec);
//     println!("Attention Mask: {:?}", attention_mask_vec);
//     println!("{}", "-".repeat(20));

//     // --- 2. ENCODER EMBEDDINGS ---
//     let encoder_embeddings = model.embed(&input_ids_array, false, 0);
//     println!("\n--- 2. RUST ENCODER EMBEDDINGS ---");
//     println!("Shape: {:?}", encoder_embeddings.dim());
//     println!("First vector (first 5 values): {:?}", encoder_embeddings.slice(s![0, 0, ..5]));
//     println!("{}", "-".repeat(20));

//     // --- 3. ENCODER FINAL OUTPUT ---
//     let encoder_hidden_states = model.encoder.forward(encoder_embeddings, &encoder_mask_array)?;
//     println!("\n--- 3. RUST ENCODER FINAL OUTPUT ---");
//     println!("Shape: {:?}", encoder_hidden_states.dim());
//     println!("First vector (first 5 values): {:?}", encoder_hidden_states.slice(s![0, 0, ..5]));
//     println!("{}", "-".repeat(20));

//     let mut decoder_input_ids = vec![model.config.bos_token_id];
//     // let mut generated_tokens = vec![];

//     // We only need to check the FIRST loop iteration to find the bug.
//     for _ in 0..1 {
//         let current_len = decoder_input_ids.len();
//         let mut decoder_input_array = Array2::<f32>::zeros((batch_size, current_len));
//         for (j, &id) in decoder_input_ids.iter().enumerate() {
//             decoder_input_array[[0, j]] = id as f32;
//         }
//         let mut causal_mask = create_decoder_causal_mask(current_len);
//         println!("DEBUG: Created causal mask: {:?}", causal_mask);  // Verify it's correct
//         // --- 4. DECODER EMBEDDINGS ---
//         let past_len = 0;
//         let decoder_embeddings = model.embed(&decoder_input_array, true, past_len);
//         println!("\n--- 4. RUST DECODER EMBEDDINGS (First Step) ---");
//         println!("Shape: {:?}", decoder_embeddings.dim());
//         println!("First vector (first 5 values): {:?}", decoder_embeddings.slice(s![0, 0, ..5]));
//         println!("{}", "-".repeat(20));

//         // --- 5. DECODER FINAL OUTPUT ---
//         let decoder_output = model.decoder.forward(
//             decoder_embeddings,
//             &encoder_hidden_states,
//             Some(&causal_mask),
//             &encoder_mask_array,
//         )?;
//         println!("\n--- 5. RUST DECODER FINAL OUTPUT (First Step) ---");
//         println!("Shape: {:?}", decoder_output.dim());
//         println!("First vector (first 5 values): {:?}", decoder_output.slice(s![0, 0, ..5]));
//         println!("{}", "-".repeat(20));

//         // --- 6. FINAL LOGITS ---
//         let last_token_hidden_state = decoder_output.slice(s![0, -1, ..]);
//         let logits: Array1<f32> = model.lm_head.dot(&last_token_hidden_state);
//         println!("\n--- 6. RUST FINAL LOGITS (First Step) ---");
//         println!("Shape: {:?}", logits.dim());
//         println!("First 10 logit values: {:?}", logits.slice(s![..10]));
//         println!("{}", "-".repeat(20));

//         // For debugging, we don't need to continue the loop.
//         // The code below would run in a real generation.
//         /*
//         let next_token_id = sample_token(logits, config)?;
//         if next_token_id == model.config.eos_token_id {
//             break;
//         }
//         generated_tokens.push(next_token_id);
//         decoder_input_ids.push(next_token_id);
//         */
//     }

//     // Return a placeholder string since we are only debugging the first step.
//     return Ok("Debugging... check console output.".to_string());
// }
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
fn sample_token(mut logits: Array1<f32>, config: &GenerationConfig) -> Result<u32> {
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
