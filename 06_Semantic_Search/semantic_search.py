from sentence_transformers import SentenceTransformer, util

#Load the pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
#sentences to convert vectors
sentences = [
    "Python is a programming Language",
    "Machine learning allows computers to learn from data.",
    "Kubernetes is used to manage containerized applications.",
    "Docker packages applications into containers.",
    "Neural networks are used in deep learning."
]

#Convert sentences to embeddings
document_embeddings = model.encode(sentences)

#query to search for
#query = "How do computers learn from data?"
query= "What is used to manage containers?"

#Convert the query to an embedding
query_embedding = model.encode(query)

#Compare the query with the document embeddings
similarities = util.cos_sim(query_embedding, document_embeddings)[0]

#Find the highest similarity score 
best_match_index = similarities.argmax().item()

print("User query:", query)
print("Best matching document:", sentences[best_match_index])
print("Similarity score:", similarities[best_match_index].item())

print("\nTop 3 matching documents:")

top_results = similarities.argsort(descending=True)[:3]

for index in top_results:
    print(sentences[index.item()])
    print("Similarity:", similarities[index].item())

