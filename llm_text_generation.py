import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

modelname="distilgpt2"
tokenizer=AutoTokenizer.from_pretrained(modelname)

prompt="Artificial intelligence will"
inputs= tokenizer(prompt,return_tensors="pt")

print("Token IDs:", inputs["input_ids"])
print("input shape:", inputs["input_ids"].shape)
print("Tokens:",tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

model=AutoModelForCausalLM.from_pretrained(modelname)
print("Model Loaded Successfully")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

print("Generated token IDs:", outputs)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
print("Generated text:")
print(generated_text)

with torch.no_grad():
    predictions=model(**inputs)

print("Logits shape:", predictions.logits.shape)

last_token_logits = predictions.logits[:, -1, :]
probabilties = torch.softmax(last_token_logits, dim=-1)

print("Logits shape:", last_token_logits.shape)
print("Probabilties shape:", probabilties.shape)
print("Total probability:", probabilties.sum())

top_probabilities, top_indices = torch.topk(probabilties, 5)
print("Top 5 probabilities:", top_probabilities)
print("Top 5 token IDs:", top_indices)

for token_id, prob in zip(top_indices[0], top_probabilities[0]):
    word = tokenizer.decode([token_id.item()])
    print(f"Token: {word!r}, Probability: {prob.item():.2%}")

greedy_next_token = torch.argmax(probabilties, dim=-1)

sampled_next_token = torch.multinomial(
    probabilties,
    num_samples=1
)

print("Greedy:", tokenizer.decode(greedy_next_token[0].item()))
print("Sampled:", tokenizer.decode(sampled_next_token[0].item()))    