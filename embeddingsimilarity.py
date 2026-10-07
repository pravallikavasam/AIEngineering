import torch
import torch.nn.functional as F

#vector_a = torch.tensor([1.0,2.0,3.0])
#vector_b = torch.tensor([1.0,2.0,3.0])
#vector_c = torch.tensor([-1.0,-2.0,-3.0])

vector_a = torch.tensor([1.0,2.0,3.0])
vector_b = torch.tensor([1.0,2.0,2.0])
vector_c = torch.tensor([3.0,-2.0,1.0])

similarity_ab=F.cosine_similarity(vector_a,vector_b,dim=0)
similarity_ac=F.cosine_similarity(vector_a,vector_c,dim=0)

print("Similarity A and B")
print(similarity_ab)
print("Similarity A and C")
print(similarity_ac)
