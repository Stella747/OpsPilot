from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import SessionLocal
from models import IncidentDB


import json
import ollama

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



SIMILARITY_THRESHOLD = 0.40

def has_sufficient_historical_evidence(
    best_similarity: float,
    threshold: float = SIMILARITY_THRESHOLD,
) -> bool:
    """Return whether the best historical match meets the threshold."""
    return best_similarity >= threshold

app = FastAPI()

# Load the embedding model once when the API starts
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load historical incidents once at startup
with open("data/historical_incidents.json", "r", encoding="utf-8") as file:
    historical_incidents = json.load(file)

# Convert historical incidents into searchable text
historical_incident_texts = [
    (
        f"Service: {item['service']}. "
        f"Severity: {item['severity']}. "
        f"Symptoms: {item['symptoms']}. "
        f"Root cause: {item['root_cause']}. "
        f"Resolution: {item['resolution']}. "
        f"Tags: {', '.join(item['tags'])}."
    )
    for item in historical_incidents
]

# Calculate historical embeddings once at startup
historical_embeddings = embedding_model.encode(
    historical_incident_texts
)



class Incident(BaseModel):
    id: str
    service: str
    severity: str
    description: str


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def home():
    return {"message": "OpsPilot is running!"}


@app.post("/incidents")
def investigate_incident(
    incident: Incident,
    db: Session = Depends(get_db)
):
    existing_incident = db.query(IncidentDB).filter(
        IncidentDB.id == incident.id
    ).first()

    if existing_incident:
        raise HTTPException(
            status_code=409,
            detail=f"Incident {incident.id} already exists"
        )

    new_incident = IncidentDB(
        id=incident.id,
        service=incident.service,
        severity=incident.severity,
        description=incident.description
    )

    db.add(new_incident)
    db.commit()

    return {
        "message": "Incident received and saved",
        "incident_id": incident.id,
        "service": incident.service,
        "severity": incident.severity,
        "status": "investigating"
    }

@app.get("/incidents")
def get_incidents(db: Session = Depends(get_db)):
    incidents = db.query(IncidentDB).all()

    return incidents

@app.get("/incidents/{incident_id}")
def get_incident(
    incident_id: str,
    db: Session = Depends(get_db)
):
    incident = db.query(IncidentDB).filter(
        IncidentDB.id == incident_id
    ).first()

    if not incident:
        raise HTTPException(
            status_code=404,
            detail=f"Incident {incident_id} not found"
        )

    return incident


@app.post("/incidents/{incident_id}/investigate")
def investigate_incident_with_ai(
    incident_id: str,
    db: Session = Depends(get_db)
):
    # 1. Retrieve the incident from SQLite
    incident = db.query(IncidentDB).filter(
        IncidentDB.id == incident_id
    ).first()

    if not incident:
        raise HTTPException(
            status_code=404,
            detail=f"Incident {incident_id} not found"
        )

    # 2. Load application logs
    with open("data/application_logs.json", "r") as file:
        application_logs = json.load(file)

    important_logs = [
        log for log in application_logs
        if log["service"] == incident.service
        and log["level"] in ["ERROR", "WARNING"]
    ]

    # 4. Find similar historical incidents using cached embeddings
    model = embedding_model

    incident_texts = [
        (
            f"Service: {item['service']}. "
            f"Severity: {item['severity']}. "
            f"Symptoms: {item['symptoms']}. "
            f"Root cause: {item['root_cause']}. "
            f"Resolution: {item['resolution']}. "
            f"Tags: {', '.join(item['tags'])}."
        )
        for item in historical_incidents
    ]

    incident_embeddings = model.encode(incident_texts)

    query = (
        f"Service: {incident.service}. "
        f"Severity: {incident.severity}. "
        f"Description: {incident.description}."
    )

    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        historical_embeddings
    )[0]

    ranked_incidents = sorted(
        zip(historical_incidents, similarities),
        key=lambda item: item[1],
        reverse=True
    )

    top_matches = [
        {
            "incident_id": item["incident_id"],
            "similarity": round(float(score), 4),
            "root_cause": item["root_cause"],
            "resolution": item["resolution"]
        }
        for item, score in ranked_incidents[:3]
    ]
    # Check whether the best historical match meets the threshold
    best_similarity = top_matches[0]["similarity"] if top_matches else 0.0

    #stop historical based recommendations when evidence is weak
    sufficient_historical_evidence= (has_sufficient_historical_evidence(best_similarity,SIMILARITY_THRESHOLD,))
    if not sufficient_historical_evidence:
        return {
            "incident_id": incident.id,
            "service": incident.service,
            "severity": incident.severity,
            "important_logs_count": len(important_logs),
            "similar_incidents": top_matches,
            "investigation_report": (
                "Insufficient historical evidence. "
                "The closest historical incident did not meet "
                "the configured similarity threshold. "
                "Do not assume it's root casue or resolution applies. "
                "Investigate the current incident using it's own logs, "
                "system metrics, and service health information."
            ),
            "best_similarity": best_similarity,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "sufficient_historical_evidence": False,
        }



    # 5. Prepare evidence for Qwen
    evidence = {
        "incident": {
            "id": incident.id,
            "service": incident.service,
            "severity": incident.severity,
            "description": incident.description
        },
        "application_logs": [
            {
                "timestamp": log["timestamp"],
                "level": log["level"],
                "message": log["message"]
            }
            for log in important_logs
        ],
        "similar_historical_incidents": top_matches
    }

    prompt = f"""
You are OpsPilot, an IT incident investigation assistant.

Analyze the evidence below:

{json.dumps(evidence, indent=2)}

Produce a report with:
1. Incident summary
2. Observed evidence
3. Most likely hypothesis
4. Recommended investigation steps
5. Confidence and limitations

Rules:
- Separate observed facts from hypotheses.
- Do not claim the root cause is confirmed.
- Use historical similarity as supporting evidence, not proof.
- Do not invent facts or metrics.
- Do not claim to have executed remediation.
- Recommend validation before production changes.
- Use only timestamps explicitly supplied in the evidence.
- Never invent timestamps, metrics, or incident details.
- If information is missing, state that it is unavailable.
- If historical evidence is insufficient, explicitly say so.
- Do not recommend a historical resolution as applicable when the evidence is weak.
- Distinguish observed logs from historical similarities and hypotheses.
"""

    # 6. Generate the report with the local model
    try:
        response = ollama.chat(
            model="qwen2.5:1.5b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Local AI model is unavailable: {exc}"
        ) from exc

    return {
        "incident_id": incident.id,
        "service": incident.service,
        "severity": incident.severity,
        "important_logs_count": len(important_logs),
        "similar_incidents": top_matches,
        "investigation_report": response["message"]["content"],
        "best_similarity": best_similarity,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "sufficient_historical_evidence": sufficient_historical_evidence,
    }
