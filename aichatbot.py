import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

modelname= "google/flan-t5-small"

tokenizer=AutoTokenizer.from_pretrained(modelname)
model=AutoModelForSeq2SeqLM.from_pretrained(modelname)

model.eval()

print("Chatbot model loaded successfully")

while True:
    question = input("You: ")

    if question.strip().lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Goodbye!")
        break
    prompt = f"Respond to the following message in one short sentence: {question}"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=False
            )

    answer = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
    )
    print("Question:", question)
    print("Chatbot:", answer)

    