import torch
import torch.nn as nn
torch.manual_seed(42)

#Define Trasformer Class
class TransformerBlock(nn.Module):
   # Constructor: initialize layers
    def __init__(self, embed_dim,num_heads):
        super().__init__()

        #Create multi-head self-attention
        self.attention=nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        #Create nomalization layer
        self.norm1=nn.LayerNorm(embed_dim)
        self.norm2=nn.LayerNorm(embed_dim)
        
        # Feed-forward neural network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, embed_dim)
        )

        # Define how input passes through the Transformer
    def forward(self, x):

        # Step 1: Multi-head self-attention
        attention_output, _ = self.attention(x, x, x)

        # Step 2: Residual connection + normalization
        x = self.norm1(x + attention_output)

        # Step 3: Feed-forward neural network
        ff_output = self.feed_forward(x)

        # Step 4: Residual connection + normalization
        x = self.norm2(x + ff_output)

        return x

 
# Input: 1 batch, 3 tokens, 4 features per token
X = torch.tensor([[
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0]
]])


# Create Transformer object
model = TransformerBlock(embed_dim=4, num_heads=2)

# Pass input through Transformer
output = model(X)

# Print results
print("Input shape:", X.shape)
print("Output shape:", output.shape)
print("Transformer output:")
print(output)       