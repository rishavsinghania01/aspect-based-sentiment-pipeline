import pandas as pd
import pytest

from aspect_sentiment_pipeline.data import prepare_reviews


def test_rejects_ratings_outside_the_supported_scale():
    frame = pd.DataFrame(
        {
            "review_text": [f"Review {index}" for index in range(8)],
            "rating": [1, 2, 3, 4, 5, 6, 2, 3],
        }
    )

    with pytest.raises(ValueError, match="between 1 and 5"):
        prepare_reviews(frame)


def test_rejects_non_numeric_ratings():
    frame = pd.DataFrame(
        {
            "review_text": [f"Review {index}" for index in range(8)],
            "rating": [1, 2, 3, 4, 5, "bad", 2, 3],
        }
    )

    with pytest.raises(ValueError, match="must be numeric"):
        prepare_reviews(frame)
