from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sentence= "I love Artificial Intelligence"
words=sentence.split()
print("Original Sentence:")
print(sentence)
print("words:")
print(words)
print("Number of words")
print(len(words))
tokens = tokenizer.tokenize(sentence)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Token IDs:")
print(token_ids)
print("Tokens:")
print(tokens)
