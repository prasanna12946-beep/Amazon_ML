import pandas as pd

from src.normalize import normalize_dataframe
from src.blocking import (
    exact_name_candidates,
    first_token_candidates,
    soundex_candidates,
    postal_candidates,
    tfidf_candidates,
    combine_candidates,
)

# Take 500 real S1 records from ground truth
gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t",
    dtype=str
).fillna("").head(500)

# Load source files
s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    dtype=str
).fillna("")

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    dtype=str
).fillna("")

s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    dtype=str
).fillna("")


# Get the S1 records used in ground truth
s1_ids = set(gt["source1_entity_id"])

s1_sample = s1[
    s1["entity_id"].isin(s1_ids)
].copy()


# Extract actual matched S2/S3 IDs
matched_ids = set()

for value in gt["matched_entity_ids"]:
    if value:
        matched_ids.update(value.split(","))


s2_ids = {
    x for x in matched_ids
    if x.startswith("S2-")
}

s3_ids = {
    x for x in matched_ids
    if x.startswith("S3-")
}


s2_sample = s2[
    s2["entity_id"].isin(s2_ids)
].copy()

s3_sample = s3[
    s3["entity_id"].isin(s3_ids)
].copy()


print("S1 records:", len(s1_sample))
print("True S2 matches:", len(s2_sample))
print("True S3 matches:", len(s3_sample))


# Normalize
print("Normalizing...")

n1 = normalize_dataframe(s1_sample)
n2 = normalize_dataframe(s2_sample)
n3 = normalize_dataframe(s3_sample)


# Fast blockers only
print("Running blockers...")

exact2 = exact_name_candidates(n1, n2)
first2 = first_token_candidates(n1, n2)
sound2 = soundex_candidates(n1, n2)
postal2 = postal_candidates(n1, n2)
tfidf2 = tfidf_candidates(n1, n2, top_k=100)

exact3 = exact_name_candidates(n1, n3)
first3 = first_token_candidates(n1, n3)
sound3 = soundex_candidates(n1, n3)
postal3 = postal_candidates(n1, n3)
tfidf3 = tfidf_candidates(n1, n3, top_k=100)

candidates = combine_candidates(
    pd.concat([exact2, exact3]),
    pd.concat([first2, first3]),
    pd.concat([sound2, sound3]),
    pd.concat([postal2, postal3]),
    pd.concat([tfidf2, tfidf3]),
)
candidate_set = set(
    zip(
        candidates["source1_entity_id"],
        candidates["candidate_entity_id"],
    )
)


# Calculate recall
total_true = 0
captured = 0

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    if not row["matched_entity_ids"]:
        continue

    for matched_id in row["matched_entity_ids"].split(","):

        total_true += 1

        if (s1_id, matched_id) in candidate_set:
            captured += 1


recall = captured / total_true if total_true else 0

print()
print("========== RECALL RESULT ==========")
print("True pairs:", total_true)
print("Captured:", captured)
print(f"Recall: {recall:.4%}")
print("===================================")