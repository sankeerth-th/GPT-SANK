import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1) Check MPS or fall back to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 2) Choose a small GPT-2 model (e.g. 'gpt2', ~124M parameters)
model_name = "gpt2"

# 3) Load tokenizer and model from Hugging Face
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# 4) Define a prompt
prompt = "In a shocking discovery, scientists revealed that"

# 5) Convert prompt to model input IDs
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# 6) Generate text
#    - You can experiment with different parameters for creativity (temperature, top_k, top_p)
with torch.no_grad():
    output_tokens = model.generate(
        input_ids, 
        max_length=50,          # total length of generated text (prompt + new tokens)
        do_sample=True,         # enable sampling (more creative)
        top_k=50,               # consider only top 50 tokens at each step
        top_p=0.95,             # or nucleus sampling
        temperature=1.0,        # 1.0 is baseline; higher = more random
        repetition_penalty=1.1  # tweak to reduce repeated phrases
    )

# 7) Decode the generated tokens back into text
generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print("\n--- Generated Text ---\n")
print(generated_text)

print("\n--- Done! ---")
