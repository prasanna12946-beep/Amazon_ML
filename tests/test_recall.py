import pandas as pd

from src.normalize import normalize_dataframe
from src.blocking import generate_all_candidates


# Load a manageable sample
s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    dtype=str
).fillna("").head(5000)

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    dtype=str
).fillna("").head(5000)

s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    dtype=str
).fillna("").head(5000)

print("Normalizing data...")

n1 = normalize_dataframe(s1)
n2 = normalize_dataframe(s2)
n3 = normalize_dataframe(s3)

print("Generating candidates...")

candidates = generate_all_candidates(n1, n2, n3)

print("Candidate pairs:", len(candidates))
print(candidates.head(10))