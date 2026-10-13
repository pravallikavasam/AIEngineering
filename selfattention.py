import torch
import math 
# 3 tokens, each represented by 2 features
X=torch.tensor([
    [1.0,0.0],
    [0.0,1.0],
    [1.0,1.0]
])

# Query, Key, Value
Q = X
K = X
V = X

# Calculate attention scores
scores = Q @ K.T

# Scale scores
d_k = K.shape[-1]
scaled_scores = scores / math.sqrt(d_k)

# Convert scores into attention weights
weights = torch.softmax(scaled_scores, dim=-1)

# Calculate attention output
output = weights @ V

print("Input X:")
print(X)

print("Attention scores:")
print(scores)

print("Scaled scores:")
print(scaled_scores)

print("Attention weights:")
print(weights)

print("Attention output:")
print(output)

print("Output shape:")
print(output.shape)