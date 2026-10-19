from sentence_transformers import SentenceTransformer
from sentence_transformers import util

#Load the pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu') 
#sentences to convert vectors
sentences = [
    "I love learning python",
    "I enjoy studying python programming",
    "The weather is hot today"
]

#GEnerate embeddings for the sentences
embeddings = model.encode(sentences)

print("number of sentences:", len(sentences))
print("Emdeddings shape:", embeddings.shape)

print("\nFirst sentence:", sentences[0])
print("First 10 embedding values:")
print(embeddings[0][:10])

similarity_1_2 = util.cos_sim(embeddings[0], embeddings[1])
similarity_1_3 = util.cos_sim(embeddings[0], embeddings[2])

print("\nCosine similarity between sentence 1 and 2:", similarity_1_2.item())
print("Cosine similarity between sentence 1 and 3:", similarity_1_3.item())