import numpy as np

sentence1= np.array([0.8,0.2,0.6])
sentence2= np.array([0.7,0.3,0.5])

dot_product= np.dot(sentence1,sentence2)

magnitude1= np.linalg.norm(sentence1)
magnitude2= np.linalg.norm(sentence2)

similarity= dot_product/(magnitude1*magnitude2)

print("Dot product:", dot_product)
print("Magnitude of sentence 1:", magnitude1)
print("Magnitude of sentence 2:", magnitude2)
print("Cosine similarity:", similarity)


