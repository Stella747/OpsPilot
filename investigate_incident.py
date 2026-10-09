import json

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from database import SessionLocal
from models import IncidentDB


# --------------------------------------------------
# 1. Get an incident from the SQLite database
# --------------------------------------------------

db = SessionLocal()

incident = db.query(IncidentDB).filter(
    IncidentDB.id == "INC-002"
).first()

db.close()


if not incident:
    print("Incident not found.")
    exit()


print("Incident being investigated:")
print(f"ID: {incident.id}")
print(f"Service: {incident.service}")
print(f"Severity: {incident.severity}")
print(f"Description: {incident.description}")
print()

# Load application logs
with open("data/application_logs.json", "r") as file:
    application_logs = json.load(file)


# Find important logs for the affected service
important_logs = []

for log in application_logs:
    if (
        log["service"] == incident.service
        and log["level"] in ["ERROR", "WARNING"]
    ):
        important_logs.append(log)


print("Important application logs:")
print()

for log in important_logs:
    print(f"[{log['level']}] {log['message']}")

print()



# --------------------------------------------------
# 2. Load historical incidents
# --------------------------------------------------

with open("data/historical_incidents.json", "r") as file:
    historical_incidents = json.load(file)


# --------------------------------------------------
# 3. Load embedding model
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 4. Create searchable text for historical incidents
# --------------------------------------------------

incident_texts = []

for historical_incident in historical_incidents:

    text = (
        f"Service: {historical_incident['service']}. "
        f"Severity: {historical_incident['severity']}. "
        f"Symptoms: {historical_incident['symptoms']}. "
        f"Root cause: {historical_incident['root_cause']}. "
        f"Resolution: {historical_incident['resolution']}. "
        f"Tags: {', '.join(historical_incident['tags'])}."
    )

    incident_texts.append(text)


# --------------------------------------------------
# 5. Create embeddings for historical incidents
# --------------------------------------------------

incident_embeddings = model.encode(incident_texts)


# --------------------------------------------------
# 6. Create a query from the current incident
# --------------------------------------------------

query = (
    f"Service: {incident.service}. "
    f"Severity: {incident.severity}. "
    f"Description: {incident.description}."
)

query_embedding = model.encode([query])


# --------------------------------------------------
# 7. Calculate similarity
# --------------------------------------------------

similarities = cosine_similarity(
    query_embedding,
    incident_embeddings
)[0]


# --------------------------------------------------
# 8. Rank historical incidents
# --------------------------------------------------

results = []

for historical_incident, similarity in zip(
    historical_incidents,
    similarities
):
    results.append(
        (historical_incident, similarity)
    )


results.sort(
    key=lambda item: item[1],
    reverse=True
)


# --------------------------------------------------
# 9. Display investigation results
# --------------------------------------------------

print("Most similar historical incidents:")
print()

for historical_incident, similarity in results[:3]:

    print(
        f"ID: {historical_incident['incident_id']}"
    )

    print(
        f"Similarity: {similarity:.4f}"
    )

    print(
        f"Root Cause: {historical_incident['root_cause']}"
    )

    print(
        f"Resolution: {historical_incident['resolution']}"
    )

    print("-" * 60)



# Build an evidence-based investigation summary
print()
print("=" * 60)
print("OPSPILOT INVESTIGATION SUMMARY")
print("=" * 60)

error_messages = [
    log["message"]
    for log in important_logs
    if log["level"] == "ERROR"
]

print(f"Incident: {incident.id}")
print(f"Service: {incident.service}")
print(f"Severity: {incident.severity}")
print()

print("Observed evidence:")

for message in error_messages:
    print(f"- {message}")

if results:
    best_match = results[0][0]

    print()
    print("Historical comparison:")
    print(f"Closest incident: {best_match['incident_id']}")
    print(f"Historical root cause: {best_match['root_cause']}")
    print(f"Historical resolution: {best_match['resolution']}")

    print()
    print(
        "Assessment: The historical incident provides a "
        "possible explanation, not a confirmed root cause."
    )

