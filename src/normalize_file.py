import pandas as pd
from .normalize import normalize_dataframe


def normalize_file(input_path, output_path, chunk_size=100000):
    """
    Normalize a large TSV file in chunks.
    """

    first_chunk = True

    for chunk in pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        chunksize=chunk_size,
    ):
        chunk = chunk.fillna("")

        normalized_chunk = normalize_dataframe(chunk)

        normalized_chunk.to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )

        first_chunk = False

        print(
            f"Processed {len(chunk):,} rows"
        )