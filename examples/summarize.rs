//! BART summarization example
use anyhow::Result;
use edgegpt::{GenerativeModel, ModelType, GenerationConfig};

fn main() -> Result<()> {
    println!("Loading DistilBART model for summarization...");
    
    let model = GenerativeModel::from_pretrained(ModelType::DistilBartCnn12_6)?;

    let article_text = "The Apollo 11 mission was the first manned mission to land on the Moon. \
    The mission, carried out by NASA, took place in July 1969. The three astronauts \
    on board were Neil Armstrong, Buzz Aldrin, and Michael Collins. Armstrong was the \
    first person to step onto the lunar surface, famously declaring, 'That's one small \
    step for man, one giant leap for mankind.' The mission successfully collected lunar \
    samples and returned the astronauts safely to Earth, marking a significant milestone \
    in human space exploration.";

    let config = GenerationConfig {
    max_new_tokens: 60,
    temperature: 1.0,        // Higher temperature for more diversity
    top_k: Some(1),
    top_p: Some(0.9),       // Nucleus sampling
    repetition_penalty: 1.2, // Stronger penalty against repetition
    sampling_strategy: edgegpt::SamplingStrategy::TopKTopP,
    ..Default::default()
    };
    
    println!("\nOriginal Text:\n{}", article_text);

    let summary = model.generate(article_text, &config)?;

    println!("\nGenerated Summary:\n{}", summary);
    
    Ok(())
}