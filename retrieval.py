from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sentence_transformers import SentenceTransformer


LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    """
    Load the local embedding model once.

    The first run downloads the model.
    Later runs use the cached copy.
    """

    return SentenceTransformer(
        LOCAL_EMBEDDING_MODEL
    )


def build_vector_database(
    records: list[dict[str, Any]],
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Create local embeddings for document chunks."""

    if not records:
        return []

    model = load_embedding_model()

    texts = [
        record["text"]
        for record in records
    ]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    vector_records = []
    total = len(records)

    for index, (record, embedding) in enumerate(
        zip(records, embeddings),
        start=1,
    ):
        vector_records.append(
            {
                **record,
                "embedding": embedding.tolist(),
            }
        )

        if progress_callback:
            progress_callback(
                index,
                total,
            )

    return vector_records


def create_embedding_matrix(
    vector_database: list[dict[str, Any]],
) -> np.ndarray | None:
    """Create a NumPy matrix for fast searching."""

    if not vector_database:
        return None

    return np.asarray(
        [
            record["embedding"]
            for record in vector_database
        ],
        dtype=np.float32,
    )


def search_documents(
    question: str,
    vector_database: list[dict[str, Any]],
    embedding_matrix: np.ndarray,
    top_k: int = 4,
    minimum_score: float = 0.15,
) -> list[dict[str, Any]]:
    """Find the most relevant document chunks."""

    if (
        not vector_database
        or embedding_matrix is None
    ):
        return []

    model = load_embedding_model()

    question_embedding = model.encode(
        question,
        normalize_embeddings=True,
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype=np.float32,
    )

    scores = (
        embedding_matrix
        @ question_embedding
    )

    ranked_indices = np.argsort(
        scores
    )[::-1]

    results = []

    for index in ranked_indices:
        score = float(scores[index])

        if score < minimum_score:
            continue

        result = vector_database[
            int(index)
        ].copy()

        result["similarity"] = score
        results.append(result)

        if len(results) >= top_k:
            break

    return results


def save_vector_database(
    vector_database: list[dict[str, Any]],
    file_path: str | Path,
) -> None:
    """Save the vector database locally."""

    output_path = Path(file_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            vector_database,
            file,
        )


def load_vector_database(
    file_path: str | Path,
) -> list[dict[str, Any]]:
    """Load the saved vector database."""

    input_path = Path(file_path)

    if not input_path.exists():
        return []

    try:
        with input_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return []