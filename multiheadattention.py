import torch
import torch.nn as nn
torch.manual_seed(42)

#3 Tokens each with 4 features
X=torch.tensor([
    [1.0,0.0,1.0,0.0],
    [0.0,1.0,0.0,1.0],
    [1.0,1.0,0.0,0.0]
])

attention= nn.MultiheadAttention(
    embed_dim=4,
    num_heads=2,
    batch_first=True    
)

# Add batch dimension
X = X.unsqueeze(0)

output,weights=attention(
    X,X,X,
    need_weights=True,
    average_attn_weights=False
)

print("Input shape:", X.shape)
print("Attention output shape:", output.shape)
print("Attention weights shape:", weights.shape)

print("Head 1 weights:")
print(weights[0, 0])

print("Head 2 weights:")
print(weights[0, 1])

print("Final output:")
print(output)



#embed_dim=4	Each token has 4 features
#num_heads=2	Two attention heads
#4 / 2 = 2	Each head works with 2-dimensional Q, K and V
#batch_first=True	Input format is batch × tokens × features
#unsqueeze(0)	Adds a batch dimension
#average_attn_weights=False	Returns separate attention weights for both heads