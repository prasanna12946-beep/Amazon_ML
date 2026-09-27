import pandas as pd

from src.normalize import normalize_dataframe
from src.blocking import generate_all_candidates, format_candidate_output


print("Loading test data...")

s1 = pd.read_csv(
    "dataset/test/test_source1.tsv",
    sep="\t",
    dtype=str
).fillna("")

s2 = pd.read_csv(
    "dataset/test/test_source2.tsv",
    sep="\t",
    dtype=str
).fillna("")

s3 = pd.read_csv(
    "dataset/test/test_source3.tsv",
    sep="\t",
    dtype=str
).fillna("")

print("S1:", len(s1))
print("S2:", len(s2))
print("S3:", len(s3))

print("Normalizing...")

n1 = normalize_dataframe(s1)
n2 = normalize_dataframe(s2)
n3 = normalize_dataframe(s3)

print("Generating candidates...")

candidates = generate_all_candidates(
    n1,
    n2,
    n3
)

print("Internal candidate pairs:", len(candidates))

output = format_candidate_output(
    candidates,
    n1
)

output.to_csv(
    "output/candidate_pairs.tsv",
    sep="\t",
    index=False
)

print("Saved: output/candidate_pairs.tsv")
print("Output rows:", len(output))
print("Expected S1 rows:", len(n1))