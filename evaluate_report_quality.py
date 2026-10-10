
import json
import re
from pathlib import Path

import ollama


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_NAME = "qwen2.5:1.5b"


# --------------------------------------------------
# 1. Load test evidence
# --------------------------------------------------

with (DATA_DIR / "application_logs.json").open(
    "r", encoding="utf-8"
) as file:
    all_logs = json.load(file)

with (DATA_DIR / "historical_incidents.json").open(
    "r", encoding="utf-8"
) as file:
    historical_incidents = json.load(file)


incident = {
    "id": "EVAL-001",
    "service": "Payment API",
    "severity": "High",
    "description": (
        "Payment API latency is increasing and requests "
        "are timing out."
    ),
}

# Only supply logs for the relevant service.
relevant_logs = [
    log for log in all_logs
    if log["service"] == incident["service"]
    and log["level"] in ["ERROR", "WARNING"]
]

# Use two known historical cases as supporting evidence.
historical_evidence = [
    item for item in historical_incidents
    if item["incident_id"] in ["HIST-001", "HIST-002"]
]

evidence = {
    "incident": incident,
    "application_logs": relevant_logs,
    "similar_historical_incidents": historical_evidence,
}


# --------------------------------------------------
# 2. Generate the investigation report
# --------------------------------------------------

prompt = f"""
You are OpsPilot, an IT incident investigation assistant.

Analyze this evidence:
{json.dumps(evidence, indent=2)}

Write a report with these five sections:
1. Incident summary
2. Observed evidence
3. Most likely hypothesis
4. Recommended investigation steps
5. Confidence and limitations

Rules:
- Separate observed facts from hypotheses.
- Do not claim the root cause is confirmed.
- Historical cases are supporting evidence, not proof.
- Do not invent logs, timestamps, metrics, or actions.
- Do not claim that you executed remediation.
- Recommend validation before production changes.
- Explicitly identify missing information.
"""

print("Generating report with local Ollama model...")
response = ollama.chat(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": prompt}],
)

report = response["message"]["content"]

print("\n" + "=" * 72)
print("GENERATED INVESTIGATION REPORT")
print("=" * 72)
print(report)


# --------------------------------------------------
# 3. Evaluate report quality
# --------------------------------------------------

# These are simple automated checks, not a complete
# semantic verification of every claim in the report.

checks = {}

report_lower = report.lower()

checks["incident_summary"] = (
    "incident summary" in report_lower
    or "summary" in report_lower
)

checks["observed_evidence"] = (
    "observed evidence" in report_lower
    or "evidence" in report_lower
)

checks["hypothesis_section"] = (
    "hypothesis" in report_lower
)

checks["investigation_steps"] = (
    "investigation steps" in report_lower
    or "recommended steps" in report_lower
)

checks["confidence_or_limitations"] = (
    "confidence" in report_lower
    or "limitations" in report_lower
)

checks["uses_supplied_log_evidence"] = any(
    log["message"].lower() in report_lower
    for log in relevant_logs
)

# Flag suspicious absolute claims for manual review.
risky_phrases = [
    "root cause is confirmed",
    "root cause has been confirmed",
    "i have fixed the issue",
    "i restarted the server",
    "i executed the remediation",
]

checks["no_obvious_unsupported_claim_phrase"] = not any(
    phrase in report_lower
    for phrase in risky_phrases
)


# --------------------------------------------------
# 4. Print evaluation results
# --------------------------------------------------

print("\n" + "=" * 72)
print("REPORT QUALITY EVALUATION")
print("=" * 72)

for name, passed in checks.items():
    result = "PASS" if passed else "REVIEW"
    print(f"{name}: {result}")

passed_count = sum(checks.values())
total_checks = len(checks)

print(f"\nAutomated checks passed: {passed_count}/{total_checks}")
print(
    "Important: PASS means a simple rule passed; "
    "it does not prove every AI claim is correct."
)
