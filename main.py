from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Incident(BaseModel):
    id: str
    service: str
    severity: str
    description: str


@app.get("/")
def home():
    return {"message": "OpsPilot is running!"}


@app.post("/incidents")
def investigate_incident(incident: Incident):
    return {
        "message": "Incident received",
        "incident_id": incident.id,
        "service": incident.service,
        "severity": incident.severity,
        "status": "investigating"
    }