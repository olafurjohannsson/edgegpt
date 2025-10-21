//! Basic text generation example
use anyhow::Result;
use edgegpt::{ModelType, GenerativeModel};

fn main() -> Result<()> {
    let model = GenerativeModel::from_pretrained(ModelType::DistilGPT2)?;

    model.generate(
        "Once upon a time in a land far, far away,",
        &edgegpt::generation::GenerationConfig {
            max_new_tokens: 50,
            temperature: 0.7,
            ..Default::default()
        },
    ).map(|output| {
        println!("Generated Text:\n{}", output);
    })?;


    
    Ok(())
}