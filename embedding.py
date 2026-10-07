import torch
import torch.nn as nn
embedding=nn.Embedding(10,3) #There are 10 possible token IDs, Every token gets represented by 3 numbers.
#token_id=torch.tensor([2])   #Give me the embedding vector stored at index 2.
token_id=torch.tensor([2,5,7]) 
output=embedding(token_id)
print("Token_ID:")
print(token_id)
print("Embedding")
print(output)
print("Embedding shape:")
print(output.shape)
