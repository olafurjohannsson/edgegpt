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

type LayerCache = (Array4<f32>, Array4<f32>);
// The full cache for the decoder is a Vec of these self-attention caches.
type FullCache = Vec<LayerCache>;
#[derive(Clone)]
struct BeamHypothesis {
    /// The sequence of tokens generated so far.
    tokens: Vec<u32>,
    /// The cumulative log probability of this sequence.
    score: f32,
    /// The KV cache for this specific beam.
    cache: Option<FullCache>,
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

pub fn generate_encoder_decoder(
    model: &BartModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    config: &GenerationConfig,
) -> Result<String> {
    println!("generate_encoder_decoder with Beam Search");

    let num_beams = 2; // You set this to 2, which is good
    let min_length = 56;
    let max_length = config.max_new_tokens;
    let eos_token_id = model.config.eos_token_id;
    let length_penalty = 1.0; // Standard value, you can tune this

    // --- Encoder Pass (This part is correct now) ---
    let input_encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let mut input_ids_vec = Vec::with_capacity(input_encoding.len() + 2);
    input_ids_vec.push(model.config.bos_token_id);
    input_ids_vec.extend(input_encoding.get_ids());
    input_ids_vec.push(eos_token_id);
    let mut input_ids_array = Array2::<f32>::zeros((1, input_ids_vec.len()));
    for (j, &id) in input_ids_vec.iter().enumerate() {
        input_ids_array[[0, j]] = id as f32;
    }
    let encoder_mask_array = Array2::<f32>::ones((1, input_ids_vec.len()));
    let encoder_embeddings = model.embed(&input_ids_array, false, 0);
    let encoder_hidden_states = model
        .encoder
        .forward(encoder_embeddings, &encoder_mask_array)?;

    // --- Initialize Beams (This part is correct now) ---
    let mut beams: Vec<BeamHypothesis> = vec![BeamHypothesis {
        tokens: vec![model.config.bos_token_id],
        score: 0.0,
        cache: None,
    }];
    let mut completed_beams: Vec<BeamHypothesis> = Vec::new();

    // --- Beam Search Loop ---
    for _ in 0..max_length {
        if beams.is_empty() {
            break;
        }
        let mut all_candidates: Vec<BeamHypothesis> = Vec::new();

        for hypo in &beams {
            let (mut logits, present_cache) =
                run_decoder_step(model, hypo, &encoder_hidden_states, &encoder_mask_array)?;

            logits = apply_repetition_penalty(logits, &hypo.tokens, config.repetition_penalty);
            let log_probs = log_softmax_1d(&logits);

            let mut top_candidates: Vec<(u32, f32)> = log_probs
                .iter()
                .enumerate()
                .map(|(id, &lp)| (id as u32, lp))
                .collect();
            top_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            // The "k" in top-k is the beam width for beam search
            top_candidates.truncate(num_beams);

            for (token_id, token_log_prob) in top_candidates {
                let mut new_tokens = hypo.tokens.clone();
                new_tokens.push(token_id);
                all_candidates.push(BeamHypothesis {
                    tokens: new_tokens,
                    score: hypo.score + token_log_prob,
                    cache: Some(present_cache.clone()),
                });
            }
        }

        all_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        beams.clear();
        for candidate in all_candidates {
            if candidate.tokens.last().cloned() == Some(eos_token_id) {
                if candidate.tokens.len() - 1 >= min_length {
                    completed_beams.push(candidate);
                }
            } else {
                beams.push(candidate);
            }
            if beams.len() == num_beams {
                break;
            }
        }
    }

    // --- 4. Select Best Hypothesis with Length Penalty ---
    let mut final_hypotheses = if completed_beams.is_empty() {
        beams
    } else {
        completed_beams
    };

    // Apply length penalty to find the best hypothesis
    final_hypotheses.sort_by(|a, b| {
        let score_a = a.score / (a.tokens.len() as f32).powf(length_penalty);
        let score_b = b.score / (b.tokens.len() as f32).powf(length_penalty);
        score_b.partial_cmp(&score_a).unwrap()
    });

    let best_hypo = final_hypotheses
        .first()
        .ok_or_else(|| anyhow::anyhow!("No hypothesis generated"))?;

    let generated_tokens = &best_hypo.tokens[1..]; // Exclude BOS token

    tokenizer
        .decode(generated_tokens, true)
        .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))
}

fn run_decoder_step(
    model: &BartModel,
    hypo: &BeamHypothesis,
    encoder_hidden_states: &Array3<f32>,
    encoder_mask_array: &Array2<f32>,
) -> Result<(Array1<f32>, FullCache)> {
    let last_token_id = *hypo.tokens.last().unwrap();
    let mut decoder_input_array = Array2::<f32>::zeros((1, 1));
    decoder_input_array[[0, 0]] = last_token_id as f32;

    let past_len = hypo.tokens.len() - 1;
    let decoder_embeddings = model.embed(&decoder_input_array, true, past_len);

    // --- THE FIX: Create a simple mask for the single-step decoder ---
    // The query length is 1, and the key length is the total sequence length.
    // The mask must be all ones to allow the new token to attend to everything.
    let attention_mask_shape = (1, past_len + 1);
    let causal_mask = Array2::ones(attention_mask_shape);

    let (decoder_output, present_cache) = model.decoder.forward(
        decoder_embeddings,
        encoder_hidden_states,
        Some(&causal_mask),
        encoder_mask_array,
        hypo.cache.as_ref(),
    )?;

    let last_token_hidden_state = decoder_output.slice(s![0, -1, ..]);
    let logits = model.lm_head.dot(&last_token_hidden_state);

    Ok((logits, present_cache))
}

// --- Helper Functions ---
fn create_decoder_causal_mask(seq_len: usize) -> Array2<f32> {
    let mut mask = Array2::<f32>::zeros((seq_len, seq_len));
    for i in 0..seq_len {
        for j in 0..=i {
            mask[[i, j]] = 1.0;
        }
    }
    mask
}

fn log_softmax_1d(logits: &Array1<f32>) -> Array1<f32> {
    let max_val = logits.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let scaled_logits = logits - max_val;
    let exp_sum = scaled_logits.mapv(f32::exp).sum();
    scaled_logits - exp_sum.ln()
}

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

pub fn generate_encoder_decoder1(
    model: &BartModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    config: &GenerationConfig,
) -> Result<String> {
    let input_encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

    // Manually add BOS and EOS tokens to the input sequence
    let mut input_ids_vec = Vec::with_capacity(input_encoding.len() + 2);
    input_ids_vec.push(model.config.bos_token_id); // Add BOS at the start
    input_ids_vec.extend(input_encoding.get_ids());
    input_ids_vec.push(model.config.eos_token_id); // Add EOS at the end

    let attention_mask_vec = vec![1; input_ids_vec.len()];

    let batch_size = 1;
    let seq_len = input_ids_vec.len();

    let mut input_ids_array = Array2::<f32>::zeros((batch_size, seq_len));
    let mut encoder_mask_array = Array2::<f32>::zeros((batch_size, seq_len));

    for (j, (&id, &mask_val)) in input_ids_vec
        .iter()
        .zip(attention_mask_vec.iter())
        .enumerate()
    {
        input_ids_array[[0, j]] = id as f32;
        encoder_mask_array[[0, j]] = mask_val as f32;
    }

    // --- 1. INPUT ---
    println!("--- 1. RUST INPUT ---");
    println!("Input IDs: {:?}", input_ids_vec);
    println!("Attention Mask: {:?}", attention_mask_vec);
    println!("{}", "-".repeat(20));

    // --- 2. ENCODER EMBEDDINGS ---
    let encoder_embeddings = model.embed(&input_ids_array, false, 0);
    println!("\n--- 2. RUST ENCODER EMBEDDINGS ---");
    println!("Shape: {:?}", encoder_embeddings.dim());
    println!(
        "First vector (first 5 values): {:?}",
        encoder_embeddings.slice(s![0, 0, ..5])
    );
    println!("{}", "-".repeat(20));

    // --- 3. ENCODER FINAL OUTPUT ---
    let encoder_hidden_states = model
        .encoder
        .forward(encoder_embeddings, &encoder_mask_array)?;
    println!("\n--- 3. RUST ENCODER FINAL OUTPUT ---");
    println!("Shape: {:?}", encoder_hidden_states.dim());
    println!(
        "First vector (first 5 values): {:?}",
        encoder_hidden_states.slice(s![0, 0, ..5])
    );
    println!("{}", "-".repeat(20));

    let mut decoder_input_ids = vec![model.config.bos_token_id];
    // let mut generated_tokens = vec![];
    let mut past_cache: Option<FullCache> = None;
    // We only need to check the FIRST loop iteration to find the bug.
    for _ in 0..1 {
        let current_len = decoder_input_ids.len();
        let mut decoder_input_array = Array2::<f32>::zeros((batch_size, current_len));
        for (j, &id) in decoder_input_ids.iter().enumerate() {
            decoder_input_array[[0, j]] = id as f32;
        }
        let mut causal_mask = create_decoder_causal_mask(current_len);
        println!("DEBUG: Created causal mask: {:?}", causal_mask); // Verify it's correct
                                                                   // --- 4. DECODER EMBEDDINGS ---
        let past_len = 0;
        let decoder_embeddings = model.embed(&decoder_input_array, true, past_len);
        println!("\n--- 4. RUST DECODER EMBEDDINGS (First Step) ---");
        println!("Shape: {:?}", decoder_embeddings.dim());
        println!(
            "First vector (first 5 values): {:?}",
            decoder_embeddings.slice(s![0, 0, ..5])
        );
        println!("{}", "-".repeat(20));

        // --- 5. DECODER FINAL OUTPUT ---
        let causal_mask = create_decoder_causal_mask(0 + 1);

        // 2. Pass the newly created causal mask to the decoder.
        let (decoder_output, present_cache) = model.decoder.forward(
            decoder_embeddings,
            &encoder_hidden_states,
            Some(&causal_mask), // <-- Pass the mask here
            &encoder_mask_array,
            past_cache.as_ref(),
        )?;
        println!("\n--- 5. RUST DECODER FINAL OUTPUT (First Step) ---");
        println!("Shape: {:?}", decoder_output.dim());
        println!(
            "First vector (first 5 values): {:?}",
            decoder_output.slice(s![0, 0, ..5])
        );
        println!("{}", "-".repeat(20));

        // --- 6. FINAL LOGITS ---
        let last_token_hidden_state = decoder_output.slice(s![0, -1, ..]);
        let logits: Array1<f32> = model.lm_head.dot(&last_token_hidden_state);
        println!("\n--- 6. RUST FINAL LOGITS (First Step) ---");
        println!("Shape: {:?}", logits.dim());
        println!("First 10 logit values: {:?}", logits.slice(s![..10]));
        println!("{}", "-".repeat(20));

        // For debugging, we don't need to continue the loop.
        // The code below would run in a real generation.
        /*
        let next_token_id = sample_token(logits, config)?;
        if next_token_id == model.config.eos_token_id {
            break;
        }
        generated_tokens.push(next_token_id);
        decoder_input_ids.push(next_token_id);
        */
    }

    // Return a placeholder string since we are only debugging the first step.
    return Ok("Debugging... check console output.".to_string());
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
