from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")


sentences = [
    "Payment API response time increased significantly.",
    "The payment service became very slow.",
    "The authentication service is failing."
]


embeddings = model.encode(sentences)


similarity_1_2 = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)[0][0]


similarity_1_3 = cosine_similarity(
    [embeddings[0]],
    [embeddings[2]]
)[0][0]


print("Similarity between payment sentences:")
print(round(similarity_1_2, 4))

print()

print("Similarity between payment and authentication sentences:")
print(round(similarity_1_3, 4))