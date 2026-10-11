---
title: OpsPilot
sdk: docker
app_port: 7860
---

# OpsPilot

OpsPilot is a portfolio demo for evidence-backed IT incident investigation. It accepts incident reports through a FastAPI API, stores them in SQLite, retrieves similar historical cases with sentence embeddings, filters weak matches, and can use an Ollama language model to draft an investigation report. The report is a hypothesis for human review; it does not confirm a root cause or perform remediation.

## Architecture and stack

- **API:** Python, FastAPI, Pydantic, and Uvicorn
- **Storage:** SQLite with SQLAlchemy
- **Retrieval:** `all-MiniLM-L6-v2` sentence embeddings and cosine similarity
- **Report generation:** Ollama with `qwen2.5:1.5b` by default; can be disabled for evidence-only mode
- **Packaging:** Docker image and Docker Compose for local development

At startup, the API loads the historical incident examples and embedding model. On investigation, it reads matching synthetic WARNING/ERROR logs, ranks historical incidents, and checks the best match against `SIMILARITY_THRESHOLD` (default `0.40`). If evidence is sufficient and `AI_PROVIDER=ollama`, it asks Ollama to draft a report. With `AI_PROVIDER=disabled`, it returns retrieval evidence without generating an LLM report.

## Data and evaluation

All incidents, logs, and retrieval evaluation records in `data/` are synthetic, fictional examples created for this demo. No production, customer, or company data is included. Their scenario labels are illustrative, not verified operational facts.

- `data/historical_incidents.json`: 15 cases across 10 fictional services.
- `data/application_logs.json`: 34 fictional timestamped log records.
- `data/retrieval_evaluation_cases.json`: 16 labeled queries, including positive examples and unrelated negative cases.
- `data/DATASET_CARD.md`: dataset scope and limitations.

Run `python evaluate_retrieval.py` to calculate Hit@1, Hit@3, MRR, and negative-case rejection at threshold 0.40. This is a small, hand-authored evaluation set and is not statistically representative; it does not establish production retrieval quality. `python evaluate_report_quality.py` applies lightweight checks to a synthetic sample report. Those checks flag items for review and do not prove factual correctness.

## Run locally with Python

Requirements: Python 3.11 or newer and a local Ollama installation for generated reports. Start Ollama and download the model in a separate terminal:

```powershell
ollama pull qwen2.5:1.5b
```

Then create an environment and start the API:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
uvicorn main:app --reload
```

The API uses the default Ollama endpoint (`http://localhost:11434`). Open `http://localhost:8000/docs` for interactive API documentation. The first API start downloads the embedding model; allow internet access and disk space. SQLite defaults to `opspilot.db` in the working directory. Set `AI_PROVIDER=disabled` to run without Ollama report generation.

## Run with Docker Compose

Docker Compose runs the API and Ollama in separate containers. It pulls the Qwen model into a named volume and persists SQLite data and embedding cache in named volumes. The first build and model downloads need internet access and several GB of disk space; CPU inference may be slow.

```powershell
docker compose up --build
```

Open `http://localhost:8000/docs` after the API starts. To stop the containers, press `Ctrl+C` and run `docker compose down`. Named volumes remain. `docker compose down -v` also removes the database and model data stored in those volumes.

This is a **local Docker deployment**, not a public cloud deployment. The Docker image uses port 7860 and defaults to `AI_PROVIDER=disabled`, which supports a hosted evidence-only demo if the image is deployed separately. Generated reports in a hosted environment require a reachable remote Ollama service and appropriate `AI_PROVIDER`, `OLLAMA_HOST`, and `OLLAMA_MODEL` settings; a hosted container cannot use Ollama running on a developer's laptop. A public hosted instance has not been created as part of this project.

## API workflow

Use `http://localhost:8000/docs` or call the API directly:

1. Submit an incident with `POST /incidents`:

   ```json
   {
     "id": "INC-1001",
     "service": "payment-api",
     "severity": "high",
     "description": "Payment requests are timing out"
   }
   ```

2. Investigate it with `POST /incidents/INC-1001/investigate`.
3. Review the returned log count, similar incidents, similarity score and threshold, evidence sufficiency, and report.

Available routes:

- `GET /` — basic health/message response
- `POST /incidents` — save a new incident (duplicate IDs return HTTP 409)
- `GET /incidents` — list saved incidents
- `GET /incidents/{incident_id}` — fetch one incident
- `POST /incidents/{incident_id}/investigate` — retrieve evidence and, when enabled and available, generate a report

## Tests

Run the test suite with:

```powershell
python -m pytest -v
```

The user reports that all six tests passed. This result has not been independently rerun in the current environment.

## Troubleshooting

- **`No module named pytest`:** activate the project virtual environment and install `requirements-dev.txt`, which includes the runtime and test dependencies.
- **Ollama unavailable (HTTP 503):** verify Ollama is running, `qwen2.5:1.5b` is pulled, and `OLLAMA_HOST`/`OLLAMA_MODEL` point to the service. In Docker Compose, wait for the model-download service to finish.
- **Slow first startup:** the embedding model downloads on first start; the initial Docker setup also downloads the Ollama model.
- **Database persistence:** local runs write `opspilot.db` in the working directory. Compose stores it in the `opspilot_data` named volume; removing volumes deletes it.
- **No report in disabled mode:** `AI_PROVIDER=disabled` intentionally returns evidence-only output.

## Current scope and limitations

OpsPilot demonstrates an incident intake and investigation workflow, similarity-based retrieval, weak-match handling, optional LLM report drafting, synthetic evaluation data, and local container packaging. It has not been validated on real incident data, and the generated reports and historical matches require human review. Logs are static synthetic examples rather than live telemetry. It does not connect to monitoring systems, execute remediation, or provide production-grade authentication or authorization. No public, always-on service is deployed by this repository setup.
