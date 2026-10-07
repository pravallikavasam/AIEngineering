from sentence_transformers import SentenceTransformer
from sentence_transformers import util
model = SentenceTransformer("all-MiniLM-L6-v2")

sentence=[
    "I love artifical intelligence",
    "I enjoy learning machine learning",
    "The weather is hot today"
]

embedding = model.encode(sentence)

print("Embedding shape")
print(embedding.shape)

similarity_01=util.cos_sim(embedding[0],embedding[1])
similarity_02=util.cos_sim(embedding[0],embedding[2])

print("AI vs machine learning")
print(similarity_01)

print("AI vs Weather")
print(similarity_02)
