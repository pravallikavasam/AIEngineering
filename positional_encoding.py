import torch
import math

# 3 tokens, 4 features each
X = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0]
])

seq_len, embed_dim = X.shape

position = torch.arange(seq_len).unsqueeze(1)
div_term = torch.exp(
    torch.arange(0, embed_dim, 2) *
    (-math.log(10000.0) / embed_dim)
)

PE = torch.zeros(seq_len, embed_dim)

PE[:, 0::2] = torch.sin(position * div_term)
PE[:, 1::2] = torch.cos(position * div_term)

output = X + PE

print("Input shape:", X.shape)
print("Positional encoding:")
print(PE)
print("Input with positional encoding:")
print(output)
print("Output shape:", output.shape)