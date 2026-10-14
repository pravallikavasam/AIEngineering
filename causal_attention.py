import torch
import torch.nn as nn

torch.manual_seed(42)

# 1 batch, 3 tokens, 4 features
X = torch.tensor([[
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0]
]])

attention = nn.MultiheadAttention(
    embed_dim=4,
    num_heads=2,
    batch_first=True
)

# Create causal attention mask
seq_len = X.shape[1]

mask = torch.triu(
    torch.ones(seq_len, seq_len, dtype=torch.bool),
    diagonal=1
)

# Apply masked self-attention
output, weights = attention(
    X, X, X,
    attn_mask=mask,
    need_weights=True,
    average_attn_weights=False
)

print("Causal mask:")
print(mask)

print("Attention weights shape:", weights.shape)

print("Head 1 weights:")
print(weights[0, 0])

print("Head 2 weights:")
print(weights[0, 1])

print("Output shape:", output.shape)