from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import SessionLocal
from models import IncidentDB


app = FastAPI()


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