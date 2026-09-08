from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .extraction import ClauseAspectExtractor


@dataclass(frozen=True)
class PreparedReviews:
    frame: pd.DataFrame
    text_column: str
    rating_column: str


@dataclass(frozen=True)
class AspectFeatures:
    values: pd.DataFrame
    mentioned: pd.DataFrame
    mentions: pd.DataFrame


def prepare_reviews(
    frame: pd.DataFrame,
    text_column: str = "review_text",
    rating_column: str = "rating",
) -> PreparedReviews:
    missing = [column for column in (text_column, rating_column) if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    prepared = frame[[text_column, rating_column]].copy()
    prepared[text_column] = prepared[text_column].fillna("").astype(str).str.strip()
    prepared[rating_column] = pd.to_numeric(prepared[rating_column], errors="coerce")
    if prepared[rating_column].isna().any():
        raise ValueError("Every rating must be numeric.")
    if not prepared[rating_column].between(1, 5, inclusive="both").all():
        raise ValueError("Ratings must be between 1 and 5.")
    prepared = prepared[prepared[text_column] != ""]
    prepared = prepared.drop_duplicates(subset=[text_column, rating_column]).reset_index(drop=True)

    if len(prepared) < 8:
        raise ValueError("At least 8 valid, distinct reviews are required for modelling.")
    if prepared[rating_column].nunique() < 2:
        raise ValueError("Ratings must contain at least two different values.")

    prepared.insert(0, "review_id", np.arange(1, len(prepared) + 1))
    return PreparedReviews(prepared, text_column, rating_column)


def build_aspect_features(
    reviews: PreparedReviews,
    extractor: ClauseAspectExtractor,
) -> AspectFeatures:
    aspects = list(extractor.aspect_lexicon)
    values = pd.DataFrame(0.0, index=reviews.frame.index, columns=aspects)
    mentioned = pd.DataFrame(False, index=reviews.frame.index, columns=aspects)
    rows: list[dict[str, object]] = []

    for row_index, row in reviews.frame.iterrows():
        extracted = extractor.extract(row[reviews.text_column])
        grouped: dict[str, list[float]] = {}
        for mention in extracted:
            grouped.setdefault(mention.aspect, []).append(mention.sentiment)
            rows.append(
                {
                    "review_id": int(row["review_id"]),
                    "aspect": mention.aspect,
                    "matched_term": mention.matched_term,
                    "sentiment": mention.sentiment,
                    "context": mention.context,
                }
            )
        for aspect, scores in grouped.items():
            values.at[row_index, aspect] = float(np.mean(scores))
            mentioned.at[row_index, aspect] = True

    mentions = pd.DataFrame(
        rows,
        columns=["review_id", "aspect", "matched_term", "sentiment", "context"],
    )
    return AspectFeatures(values=values, mentioned=mentioned, mentions=mentions)
