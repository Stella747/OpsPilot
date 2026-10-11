
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. Load historical incidents
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "historical_incidents.json"
EVALUATION_FILE = BASE_DIR / "data" / "retrieval_evaluation_cases.json"

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
# 3. Load labeled synthetic evaluation cases
# --------------------------------------------------

with EVALUATION_FILE.open("r", encoding="utf-8") as file:
    test_cases = json.load(file)



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
positive_cases = 0
negative_cases = 0
negative_rejections = 0

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
    top_score = float(scores.max())

    print(f"\nTest case: {case['name']}")
    print(f"Top similarity score: {top_score:.4f}")
    print(f"Actual top match: {ranked_ids[0]}")

    # Negative case: no expected historical match
    if expected_id is None:
        negative_cases += 1
        rejected = top_score < 0.40
        negative_rejections += int(rejected)
        print("Expected incident: No relevant historical match")
        print(f"Threshold 0.40 rejection: {'PASS' if rejected else 'REVIEW'}")
        continue

    # Positive case: an expected historical incident exists
    positive_cases += 1

    print(f"Expected incident: {expected_id}")

    if ranked_ids[0] == expected_id:
        hits_at_1 += 1

    if expected_id in ranked_ids[:3]:
        hits_at_3 += 1

    if expected_id in ranked_ids:
        rank = ranked_ids.index(expected_id) + 1
        reciprocal_ranks.append(1 / rank)
        print(f"Expected incident rank: {rank}")
    else:
        reciprocal_ranks.append(0.0)
        print("Expected incident rank: Not retrieved")

    print(f"Top 3 results: {ranked_ids[:3]}")



# --------------------------------------------------
# 6. Report aggregate metrics
# --------------------------------------------------

print("\n" + "=" * 72)
print("AGGREGATE METRICS")
print("=" * 72)

print(f"Positive test cases: {positive_cases}")
print(f"Negative test cases: {negative_cases}")

if positive_cases > 0:
    print(f"Hit@1: {hits_at_1 / positive_cases:.2%}")
    print(f"Hit@3: {hits_at_3 / positive_cases:.2%}")
    print(
        "Mean Reciprocal Rank (MRR): "
        f"{sum(reciprocal_ranks) / positive_cases:.4f}"
    )
else:
    print("No positive test cases were evaluated.")

if negative_cases:
    print(f"Negative rejection rate at 0.40: {negative_rejections / negative_cases:.2%}")

print("\nEvaluation completed.")
