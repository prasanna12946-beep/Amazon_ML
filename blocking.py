import pandas as pd


def build_exact_name_index(source1_df):
    """
    Build an index from country + normalized business name
    to Source 1 entity IDs.
    """

    index = {}

    for _, row in source1_df.iterrows():
        country = row["country_code"]
        name = row["core_name"]

        if not country or not name:
            continue

        key = (country, name)

        index.setdefault(key, []).append(row["entity_id"])

    return index


def exact_name_candidates(source1_df, source2_df):
    """
    Find Source 1 / Source 2 candidates with
    exactly matching normalized business names
    within the same country.
    """

    index = build_exact_name_index(source1_df)

    candidates = []

    for _, row in source2_df.iterrows():
        country = row["country_code"]
        name = row["core_name"]

        if not country or not name:
            continue

        key = (country, name)

        for source1_id in index.get(key, []):
            candidates.append({
                "source1_entity_id": source1_id,
                "candidate_entity_id": row["entity_id"],
                "block_signals": "exact_name",
            })

    return pd.DataFrame(
        candidates,
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
            "block_signals",
        ],

    )
def first_token_candidates(source1_df, source2_df):
    """
    Find Source 1 / Source 2 candidates that share the
    first token of the normalized business name within
    the same country.
    """

    index = {}

    for _, row in source1_df.iterrows():
        country = row["country_code"]
        name = row["core_name"]

        if not country or not name:
            continue

        first_token = name.split()[0]

        if not first_token:
            continue

        key = (country, first_token)
        index.setdefault(key, []).append(row["entity_id"])

    candidates = []

    for _, row in source2_df.iterrows():
        country = row["country_code"]
        name = row["core_name"]

        if not country or not name:
            continue

        first_token = name.split()[0]

        if not first_token:
            continue

        key = (country, first_token)

        for source1_id in index.get(key, []):
            candidates.append({
                "source1_entity_id": source1_id,
                "candidate_entity_id": row["entity_id"],
                "block_signals": "first_token",
            })

    return pd.DataFrame(
        candidates,
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
            "block_signals",
        ],
    )
def soundex(name):
    """
    Generate a simple Soundex code for the first token
    of a business name.
    """

    if not name:
        return ""

    word = name.split()[0]

    if not word:
        return ""

    word = word.upper()

    mapping = {
        "BFPV": "1",
        "CGJKQSXZ": "2",
        "DT": "3",
        "L": "4",
        "MN": "5",
        "R": "6",
    }

    code = word[0]

    previous = ""

    for char in word[1:]:
        digit = ""

        for letters, value in mapping.items():
            if char in letters:
                digit = value
                break

        if digit and digit != previous:
            code += digit

        previous = digit

    return (code + "000")[:4]


def soundex_candidates(source1_df, source2_df):
    """
    Find Source 1 / Source 2 candidates that share the
    same country and Soundex code of the first name token.
    """

    index = {}

    for _, row in source1_df.iterrows():
        country = row["country_code"]
        name = row["core_name"]

        if not country or not name:
            continue

        code = soundex(name)

        if not code:
            continue

        key = (country, code)
        index.setdefault(key, []).append(row["entity_id"])

    candidates = []

    for _, row in source2_df.iterrows():
        country = row["country_code"]
        name = row["core_name"]

        if not country or not name:
            continue

        code = soundex(name)

        if not code:
            continue

        key = (country, code)

        for source1_id in index.get(key, []):
            candidates.append({
                "source1_entity_id": source1_id,
                "candidate_entity_id": row["entity_id"],
                "block_signals": "soundex",
            })

    return pd.DataFrame(
        candidates,
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
            "block_signals",
        ],
    )
def postal_candidates(source1_df, source2_df):
    """
    Find Source 1 / Source 2 candidates that share the
    same country and postal/PIN code.
    """

    index = {}

    for _, row in source1_df.iterrows():
        country = row["country_code"]
        postal = row["postal_code"]

        if not country or not postal:
            continue

        key = (country, postal)
        index.setdefault(key, []).append(row["entity_id"])

    candidates = []

    for _, row in source2_df.iterrows():
        country = row["country_code"]
        postal = row["postal_code"]

        if not country or not postal:
            continue

        key = (country, postal)

        for source1_id in index.get(key, []):
            candidates.append({
                "source1_entity_id": source1_id,
                "candidate_entity_id": row["entity_id"],
                "block_signals": "postal",
            })

    return pd.DataFrame(
        candidates,
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
            "block_signals",
        ],
    )
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def tfidf_candidates(source1_df, source2_df, top_k=5):
    """
    Generate candidates using TF-IDF character n-gram
    similarity within the same country.
    """

    candidates = []

    for country in source1_df["country_code"].dropna().unique():

        s1_country = source1_df[
            source1_df["country_code"] == country
        ]

        s2_country = source2_df[
            source2_df["country_code"] == country
        ]

        if s1_country.empty or s2_country.empty:
            continue

        s1_names = s1_country["core_name"].fillna("")
        s2_names = s2_country["core_name"].fillna("")

        if not s1_names.str.strip().any() or not s2_names.str.strip().any():
            continue

        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=1,
        )

        combined = pd.concat(
            [s1_names, s2_names],
            ignore_index=True,
        )

        vectorizer.fit(combined)

        s1_matrix = vectorizer.transform(s1_names)
        s2_matrix = vectorizer.transform(s2_names)

        k = min(top_k, len(s1_country))

        if k == 0:
            continue

        nn = NearestNeighbors(
            n_neighbors=k,
            metric="cosine",
            algorithm="brute",
        )

        nn.fit(s1_matrix)

        distances, indices = nn.kneighbors(s2_matrix)

        s1_ids = s1_country["entity_id"].tolist()
        s2_ids = s2_country["entity_id"].tolist()

        for s2_index, s2_id in enumerate(s2_ids):

            for position in range(k):

                s1_index = indices[s2_index][position]

                candidates.append({
                    "source1_entity_id": s1_ids[s1_index],
                    "candidate_entity_id": s2_id,
                    "block_signals": "tfidf",
                })

    return pd.DataFrame(
        candidates,
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
            "block_signals",
        ],
    )
def combine_candidates(
    exact_df,
    first_token_df,
    soundex_df,
    postal_df,
    tfidf_df,
):
    """
    Combine all blocking results into one candidate set.

    Duplicate S1-S2 pairs are merged, and all blocking
    signals that generated the pair are preserved.
    """

    all_candidates = pd.concat(
        [
            exact_df,
            first_token_df,
            soundex_df,
            postal_df,
            tfidf_df,
        ],
        ignore_index=True,
    )

    if all_candidates.empty:
        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "block_signals",
            ]
        )

    combined = (
        all_candidates
        .groupby(
            [
                "source1_entity_id",
                "candidate_entity_id",
            ],
            as_index=False,
        )["block_signals"]
        .agg(lambda x: ",".join(sorted(set(x))))
    )

    return combined
def generate_all_candidates(source1_df, source2_df, source3_df):
    """
    Generate and combine blocking candidates for both
    Source 2 and Source 3 against Source 1.
    """

    def generate_for_source(source_df):
        exact = exact_name_candidates(source1_df, source_df)
        first_token = first_token_candidates(source1_df, source_df)
        soundex = soundex_candidates(source1_df, source_df)
        postal = postal_candidates(source1_df, source_df)
        tfidf = tfidf_candidates(source1_df, source_df, top_k=5)

        return combine_candidates(
            exact,
            first_token,
            soundex,
            postal,
            tfidf,
        )

    source2_candidates = generate_for_source(source2_df)
    source3_candidates = generate_for_source(source3_df)

    return pd.concat(
        [
            source2_candidates,
            source3_candidates,
        ],
        ignore_index=True,
    )
def format_candidate_output(candidates, source1_df):
    """
    Convert internal candidate pairs into the official
    candidate_pairs.tsv format.
    """

    grouped = (
        candidates
        .groupby("source1_entity_id")["candidate_entity_id"]
        .agg(lambda x: ",".join(sorted(set(x))))
        .to_dict()
    )

    output = pd.DataFrame({
        "source1_entity_id": source1_df["entity_id"],
    })

    output["candidate_entity_ids"] = (
        output["source1_entity_id"]
        .map(grouped)
        .fillna("")
    )

    return output