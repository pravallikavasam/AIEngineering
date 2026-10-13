import torch
import torch.nn as nn
import math

torch.manual_seed(42)   #Makes random initialization reproducible

# 3 tokens, 2 features each
X = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

# Learnable projections
W_q = nn.Linear(2, 2, bias=False)            #Creates a trainable linear transformation from 2 features to 2 features, Generates Query vectors, Generates Key vectors, Generates Value vectors
W_k = nn.Linear(2, 2, bias=False)
W_v = nn.Linear(2, 2, bias=False)

# Generate Query, Key and Value
Q = W_q(X)
K = W_k(X)
V = W_v(X)

# Scaled dot-product attention
scores = Q @ K.T
scaled_scores = scores / math.sqrt(K.shape[-1])
weights = torch.softmax(scaled_scores, dim=-1)
output = weights @ V  #Produces context-aware output vectors

print("Query:")
print(Q)

print("Key:")
print(K)

print("Value:")
print(V)

print("Attention weights:")
print(weights)

print("Attention output:")
print(output)

print("Output shape:")
print(output.shape)