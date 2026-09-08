from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import build_aspect_features, prepare_reviews
from .extraction import ClauseAspectExtractor, load_aspect_lexicon
from .modeling import ModelConfig, fit_rating_driver_model


@dataclass(frozen=True)
class AnalysisConfig:
    text_column: str = "review_text"
    rating_column: str = "rating"
    aspects_file: str | Path | None = None
    min_mentions: int = 3
    variance_threshold: float = 0.85
    max_components: int = 12
    test_size: float = 0.25
    random_state: int = 42
    top_review_count: int = 5


@dataclass(frozen=True)
class AnalysisArtifacts:
    summary: dict[str, Any]
    aspects: pd.DataFrame
    mentions: pd.DataFrame
    top_reviews: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": _json_safe(self.summary),
            "aspects": _json_safe(self.aspects.to_dict(orient="records")),
            "top_reviews": _json_safe(self.top_reviews.to_dict(orient="records")),
        }

    def save(self, output_directory: str | Path) -> Path:
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / "analysis.json").write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )
        self.aspects.to_csv(output / "aspect_scores.csv", index=False)
        self.mentions.to_csv(output / "aspect_mentions.csv", index=False)
        self.top_reviews.to_csv(output / "top_reviews.csv", index=False)
        return output


def analyse_reviews(
    frame: pd.DataFrame,
    config: AnalysisConfig | None = None,
) -> AnalysisArtifacts:
    settings = config or AnalysisConfig()
    reviews = prepare_reviews(frame, settings.text_column, settings.rating_column)
    lexicon = load_aspect_lexicon(settings.aspects_file)
    extractor = ClauseAspectExtractor(lexicon)
    features = build_aspect_features(reviews, extractor)

    model_config = ModelConfig(
        min_mentions=settings.min_mentions,
        variance_threshold=settings.variance_threshold,
        max_components=settings.max_components,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )
    model = fit_rating_driver_model(
        features.values,
        features.mentioned,
        reviews.frame[settings.rating_column],
        model_config,
    )

    aspects = model.aspect_statistics.copy()
    aspects["direction"] = np.where(
        aspects["rating_impact"] >= 0,
        "higher sentiment is associated with a higher rating",
        "higher sentiment is associated with a lower rating",
    )

    review_sentiment = features.values.where(features.mentioned).mean(axis=1)
    top_reviews = reviews.frame.assign(aspect_sentiment=review_sentiment).sort_values(
        [settings.rating_column, "aspect_sentiment"],
        ascending=[False, False],
        na_position="last",
    )
    top_reviews = top_reviews.head(settings.top_review_count).reset_index(drop=True)

    summary = {
        "input_rows": int(len(frame)),
        "analysed_rows": int(len(reviews.frame)),
        "average_rating": float(reviews.frame[settings.rating_column].mean()),
        "extracted_mentions": int(len(features.mentions)),
        "retained_aspects": model.retained_aspects,
        "svd_components": model.component_count,
        "explained_variance": model.explained_variance,
        "ols_intercept": model.intercept,
        "ols_r_squared": model.r_squared,
        "ols_adjusted_r_squared": model.adjusted_r_squared,
        "holdout_r_squared": model.holdout_r_squared,
        "holdout_mae": model.holdout_mae,
        "holdout_rmse": model.holdout_rmse,
        "configuration": asdict(settings),
    }
    return AnalysisArtifacts(
        summary=summary,
        aspects=aspects,
        mentions=features.mentions,
        top_reviews=top_reviews,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return None if np.isnan(numeric) or np.isinf(numeric) else numeric
    if isinstance(value, np.integer):
        return value.item()
    return value
