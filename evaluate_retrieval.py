
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. Load historical incidents
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "historical_incidents.json"

with DATA_FILE.open("r", encoding="utf-8") as file:
    historical_incidents = json.load(file)


# --------------------------------------------------
# 2. Prepare historical incident text
# --------------------------------------------------

def incident_to_text(incident):
    """Convert a historical incident into searchable text."""
    return (
        f"Service: {incident['service']}. "
        f"Severity: {incident['severity']}. "
        f"Symptoms: {incident['symptoms']}. "
        f"Root cause: {incident['root_cause']}. "
        f"Resolution: {incident['resolution']}. "
        f"Tags: {', '.join(incident['tags'])}."
    )


historical_texts = [
    incident_to_text(incident)
    for incident in historical_incidents
]


# --------------------------------------------------
# 3. Define evaluation test cases
# --------------------------------------------------

# These are synthetic queries based on known historical cases.
# Each expected ID is the case we want retrieval to find.

test_cases = [
    {
        "name": "Payment connection pool exhaustion",
        "query": (
            "Payment API is slow because database connections "
            "are exhausted and requests are waiting for connections."
        ),
        "expected_id": "HIST-001",
    },
    {
        "name": "Payment database CPU overload",
        "query": (
            "Payment requests are failing while database CPU "
            "usage is extremely high."
        ),
        "expected_id": "HIST-002",
    },
    {
        "name": "Authentication failures",
        "query": (
            "Auth API users cannot authenticate and are receiving "
            "authentication failures."
        ),
        "expected_id": "HIST-003",
    },
    {
        "name": "Order processing delays",
        "query": (
            "Order API requests are delayed because messages "
            "are accumulating in the processing queue."
        ),
        "expected_id": "HIST-004",
    },
    {
        "name": "Payment latency after deployment",
        "query": (
            "Payment API latency increased after a new software "
            "release was deployed."
        ),
        "expected_id": "HIST-005",
    },
]


# --------------------------------------------------
# 4. Generate embeddings once
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

historical_embeddings = model.encode(
    historical_texts,
    normalize_embeddings=True,
    show_progress_bar=False,
)

query_embeddings = model.encode(
    [case["query"] for case in test_cases],
    normalize_embeddings=True,
    show_progress_bar=False,
)

# Normalized embeddings allow cosine similarity comparisons.
similarity_matrix = cosine_similarity(
    query_embeddings,
    historical_embeddings,
)


# --------------------------------------------------
# 5. Evaluate retrieval quality
# --------------------------------------------------

hits_at_1 = 0
hits_at_3 = 0
reciprocal_ranks = []

print("\n" + "=" * 72)
print("OPSPILOT — SEMANTIC RETRIEVAL EVALUATION")
print("=" * 72)

for index, case in enumerate(test_cases):
    scores = similarity_matrix[index]

    ranked_indices = scores.argsort()[::-1]

    ranked_ids = [
        historical_incidents[i]["incident_id"]
        for i in ranked_indices
    ]

    expected_id = case["expected_id"]

    if ranked_ids[0] == expected_id:
        hits_at_1 += 1

    if expected_id in ranked_ids[:3]:
        hits_at_3 += 1

    if expected_id in ranked_ids:
        rank = ranked_ids.index(expected_id) + 1
        reciprocal_ranks.append(1 / rank)
    else:
        rank = None
        reciprocal_ranks.append(0.0)

    print(f"\nTest case: {case['name']}")
    print(f"Expected incident: {expected_id}")
    print(f"Actual top match:  {ranked_ids[0]}")
    print(f"Expected incident rank: {rank or 'Not retrieved'}")
    print(f"Top 3 results: {ranked_ids[:3]}")


# --------------------------------------------------
# 6. Report aggregate metrics
# --------------------------------------------------

total = len(test_cases)

hit_at_1 = hits_at_1 / total
hit_at_3 = hits_at_3 / total
mrr = sum(reciprocal_ranks) / total

print("\n" + "=" * 72)
print("AGGREGATE METRICS")
print("=" * 72)

print(f"Test cases: {total}")
print(f"Hit@1: {hit_at_1:.2%}")
print(f"Hit@3: {hit_at_3:.2%}")
print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")

print("\nEvaluation completed.")
