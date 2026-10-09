
import ollama
from database import SessionLocal
from models import IncidentDB
import json


# Retrieve the incident from the database
db = SessionLocal()

try:
    incident_record = db.query(IncidentDB).filter(
        IncidentDB.id == "INC-002"
    ).first()

    if incident_record is None:
        raise ValueError("Incident INC-002 was not found.")

    incident = {
        "id": incident_record.id,
        "service": incident_record.service,
        "severity": incident_record.severity,
        "description": incident_record.description
    }
finally:
    db.close()


# Load application logs from the JSON file

with open("data/application_logs.json", "r") as file:
    all_logs = json.load(file)


# Keep only errors and warnings for the incident's service
relevant_logs = [
    log
    for log in all_logs
    if log["service"] == incident["service"]
    and log["level"] in ["ERROR", "WARNING"]
]

# Extract log messages for the AI prompt
logs = [log["message"] for log in relevant_logs]


# Load historical incidents
with open("data/historical_incidents.json", "r") as file:
    historical_incidents = json.load(file)


# Find the historical case we want to use for this test
historical_case = next(
    (
        {
            "incident_id": item["incident_id"],
            "root_cause": item["root_cause"],
            "resolution": item["resolution"]
        }
        for item in historical_incidents
        if item["incident_id"] == "HIST-001"
    ),
    None
)

if historical_case is None:
    raise ValueError("Historical incident HIST-001 was not found.")


prompt = f"""
You are OpsPilot, an IT incident investigation assistant.

Analyze the following evidence.

CURRENT INCIDENT:
{incident}

CURRENT APPLICATION LOGS:
{logs}

SIMILAR HISTORICAL INCIDENT:
{historical_case}

Write your investigation report using these sections:
1. Incident summary
2. Observed evidence
3. Most likely hypothesis
4. Recommended investigation steps
5. Confidence and limitations

Important rules:
- Separate observed facts from hypotheses.
- Do not claim the root cause is confirmed.
- Treat the historical incident as supporting evidence, not proof.
- Do not claim that you executed any remediation.
- Recommend investigation steps before making production changes.
"""


response = ollama.chat(
    model="qwen2.5:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


print("OPSPILOT AI INVESTIGATION REPORT")
print("=" * 50)
print(response["message"]["content"])
