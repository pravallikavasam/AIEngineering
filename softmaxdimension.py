import torch

scores = torch.tensor([
    [2.0, 0.0],
    [1.0, 3.0]
])

print("Original scores:")
print(scores)

print("\nSoftmax dim=0:")
print(torch.softmax(scores, dim=0))

print("\nSoftmax dim=1:")
print(torch.softmax(scores, dim=1))