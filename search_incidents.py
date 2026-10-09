import json

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load historical incidents
with open("data/historical_incidents.json", "r") as file:
    historical_incidents = json.load(file)


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Create searchable text for every historical incident
incident_texts = []

for incident in historical_incidents:
    text = (
        f"Service: {incident['service']}. "
        f"Severity: {incident['severity']}. "
        f"Symptoms: {incident['symptoms']}. "
        f"Root cause: {incident['root_cause']}. "
        f"Resolution: {incident['resolution']}. "
        f"Tags: {', '.join(incident['tags'])}."
    )

    incident_texts.append(text)


# Convert historical incidents into embeddings
incident_embeddings = model.encode(incident_texts)


# Get search query from user
query = input("Enter your incident search: ")

print()
print(f"Searching for: {query}")
print()


# Convert the query into an embedding
query_embedding = model.encode([query])


# Calculate similarity between query and every incident
similarities = cosine_similarity(
    query_embedding,
    incident_embeddings
)[0]


# Attach similarity scores to incidents
results = []

for incident, similarity in zip(historical_incidents, similarities):
    results.append((incident, similarity))


# Sort from most similar to least similar
results.sort(
    key=lambda item: item[1],
    reverse=True
)


# Display results
for incident, similarity in results:
    print(f"ID: {incident['incident_id']}")
    print(f"Similarity: {similarity:.4f}")
    print(f"Service: {incident['service']}")
    print(f"Severity: {incident['severity']}")
    print(f"Symptoms: {incident['symptoms']}")
    print(f"Root Cause: {incident['root_cause']}")
    print(f"Resolution: {incident['resolution']}")
    print("-" * 60)