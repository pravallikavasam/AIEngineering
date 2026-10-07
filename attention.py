import torch

query=torch.tensor([1.0,0.0,1.0])

key1=torch.tensor([1.0,0.0,1.0])
key2=torch.tensor([0.0,1.0,0.0])

score1=torch.dot(query,key1)
score2=torch.dot(query,key2)

print("Score1:")
print(score1)

print("Score2:")
print(score2)

scores= torch.tensor([score1,score2])
weights=torch.softmax(scores,dim=0)

print("Attention weights")
print(weights)

value1 = torch.tensor([10.0, 0.0])
value2 = torch.tensor([0.0, 10.0])

output = weights[0] * value1 + weights[1] * value2

print("Attention output:")
print(output)