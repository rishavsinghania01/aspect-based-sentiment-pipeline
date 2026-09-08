import json
from pathlib import Path

import pandas as pd

from aspect_sentiment_pipeline.pipeline import AnalysisConfig, analyse_reviews

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_sample_pipeline_produces_model_and_export_files(tmp_path):
    frame = pd.read_csv(PROJECT_ROOT / "data" / "sample_reviews.csv")

    result = analyse_reviews(frame, AnalysisConfig(min_mentions=3))
    output = result.save(tmp_path)

    assert result.summary["analysed_rows"] == len(frame)
    assert result.summary["svd_components"] >= 1
    assert result.summary["explained_variance"] > 0
    assert result.summary["holdout_mae"] >= 0
    assert {"food", "service"}.issubset(set(result.aspects["aspect"]))
    assert result.aspects["p_value"].between(0, 1).all()
    assert not result.mentions.empty
    assert len(result.top_reviews) == 5

    expected = {
        "analysis.json",
        "aspect_scores.csv",
        "aspect_mentions.csv",
        "top_reviews.csv",
    }
    assert expected == {path.name for path in output.iterdir()}

    payload = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert payload["summary"]["average_rating"] == result.summary["average_rating"]


def test_pipeline_is_deterministic():
    frame = pd.read_csv(PROJECT_ROOT / "data" / "sample_reviews.csv")
    config = AnalysisConfig(min_mentions=3, random_state=7)

    first = analyse_reviews(frame, config)
    second = analyse_reviews(frame, config)

    pd.testing.assert_frame_equal(first.aspects, second.aspects)
    assert first.summary == second.summary
