import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# Use the exact same model and input text
model_id = "sshleifer/distilbart-cnn-12-6"
article_text = "The Apollo 11 mission was the first manned mission to land on the Moon. The mission, carried out by NASA, took place in July 1969. The three astronauts on board were Neil Armstrong, Buzz Aldrin, and Michael Collins. Armstrong was the first person to step onto the lunar surface, famously declaring, 'That's one small step for man, one giant leap for mankind.' The mission successfully collected lunar samples and returned the astronauts safely to Earth, marking a significant milestone in human space exploration."

# Set print options for readability
torch.set_printoptions(precision=6, sci_mode=False, linewidth=120)

# Load model and tokenizer
tokenizer = BartTokenizer.from_pretrained(model_id)
model = BartForConditionalGeneration.from_pretrained(model_id)
model.eval() # Set to evaluation mode
print("\n--- WEIGHT CHECKSUMS ---")
print("Encoder Layer 0 Q-Proj Weight Sum:", model.model.encoder.layers[0].self_attn.q_proj.weight.sum())
print("Encoder Layer 0 FC1 Bias Sum:", model.model.encoder.layers[0].fc1.bias.sum())
print("Decoder Layer 0 Cross-Attn V-Proj Weight Sum:", model.model.decoder.layers[0].encoder_attn.v_proj.weight.sum())
print("First Q-Proj Weight:", model.model.encoder.layers[0].self_attn.q_proj.weight[0, 0])
print("-" * 20)
print("--- 1. INPUT ---")
inputs = tokenizer(article_text, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
print(f"Input IDs: {input_ids}")
print(f"Attention Mask: {attention_mask}")
print("-" * 20)

# --- ENCODER PASS ---
print("\n--- 2. ENCODER EMBEDDINGS ---")
encoder_embeddings = model.model.encoder.embed_tokens(input_ids) + model.model.encoder.embed_positions(input_ids)
encoder_embeddings = model.model.encoder.layernorm_embedding(encoder_embeddings)
print("Shape:", encoder_embeddings.shape)
print("First vector (first 5 values):", encoder_embeddings[0, 0, :5])
print("-" * 20)

print("\n--- 3. ENCODER FINAL OUTPUT ---")
encoder_outputs = model.model.encoder(inputs_embeds=encoder_embeddings, attention_mask=attention_mask)
encoder_hidden_states = encoder_outputs.last_hidden_state
print("Shape:", encoder_hidden_states.shape)
print("First vector (first 5 values):", encoder_hidden_states[0, 0, :5])
print("-" * 20)


# --- DECODER PASS (First Step Only) ---
# We simulate the first step of our generation loop
decoder_input_ids = torch.tensor([[model.config.bos_token_id]]) # Starts with [0]

print("\n--- 4. DECODER EMBEDDINGS (First Step) ---")
decoder_embeddings = model.model.decoder.embed_tokens(decoder_input_ids) + model.model.decoder.embed_positions(decoder_input_ids)
decoder_embeddings = model.model.decoder.layernorm_embedding(decoder_embeddings)
print("Shape:", decoder_embeddings.shape)
print("First vector (first 5 values):", decoder_embeddings[0, 0, :5])
print("-" * 20)

print("\n--- 5. DECODER FINAL OUTPUT (First Step) ---")
decoder_outputs = model.model.decoder(
    inputs_embeds=decoder_embeddings,
    encoder_hidden_states=encoder_hidden_states,
    encoder_attention_mask=attention_mask
)
decoder_hidden_states = decoder_outputs.last_hidden_state
print("Shape:", decoder_hidden_states.shape)
print("First vector (first 5 values):", decoder_hidden_states[0, 0, :5])
print("-" * 20)


print("\n--- 6. FINAL LOGITS (First Step) ---")
logits = model.lm_head(decoder_hidden_states)
print("Shape:", logits.shape)
print("First 10 logit values:", logits[0, 0, :10])
print("-" * 20)
