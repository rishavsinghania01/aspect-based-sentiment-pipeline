from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from .pipeline import AnalysisConfig, analyse_reviews

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_FILE = PROJECT_ROOT / "data" / "sample_reviews.csv"


def run() -> None:
    st.set_page_config(page_title="Aspect-Based Sentiment Pipeline", layout="wide")
    st.title("Aspect-Based Sentiment Pipeline")
    st.write(
        "Upload review text with a 1 to 5 rating. The pipeline links sentiment to named "
        "aspects, reduces the sparse feature matrix with SVD, and estimates rating impact with OLS."
    )

    uploaded = st.file_uploader("Review CSV", type=["csv"])
    frame = pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(SAMPLE_FILE)
    st.caption("Using uploaded data" if uploaded is not None else "Using the included sample data")
    st.dataframe(frame.head(10), width="stretch", hide_index=True)

    available_columns = frame.columns.tolist()
    default_text = (
        available_columns.index("review_text") if "review_text" in available_columns else 0
    )
    default_rating = available_columns.index("rating") if "rating" in available_columns else 0

    first, second, third = st.columns(3)
    with first:
        text_column = st.selectbox("Review text column", available_columns, index=default_text)
    with second:
        rating_column = st.selectbox("Rating column", available_columns, index=default_rating)
    with third:
        min_mentions = st.number_input("Minimum mentions", min_value=1, value=3, step=1)

    if not st.button("Run analysis", type="primary"):
        return

    try:
        result = analyse_reviews(
            frame,
            AnalysisConfig(
                text_column=text_column,
                rating_column=rating_column,
                min_mentions=int(min_mentions),
            ),
        )
    except ValueError as error:
        st.error(str(error))
        return

    summary = result.summary
    metric_columns = st.columns(4)
    metric_columns[0].metric("Reviews analysed", summary["analysed_rows"])
    metric_columns[1].metric("Average rating", f"{summary['average_rating']:.2f}")
    metric_columns[2].metric("Aspects modelled", len(summary["retained_aspects"]))
    metric_columns[3].metric("Holdout MAE", f"{summary['holdout_mae']:.2f}")

    st.subheader("Estimated rating drivers")
    chart_data = result.aspects.set_index("aspect")["rating_impact"].sort_values()
    st.bar_chart(chart_data, horizontal=True)
    st.dataframe(
        result.aspects[
            [
                "aspect",
                "mentions",
                "mean_sentiment",
                "rating_impact",
                "p_value",
                "confidence_low",
                "confidence_high",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    st.subheader("Top positive reviews")
    st.dataframe(result.top_reviews, width="stretch", hide_index=True)

    st.subheader("Model diagnostics")
    diagnostics = {
        "SVD components": summary["svd_components"],
        "SVD explained variance": summary["explained_variance"],
        "OLS adjusted R-squared": summary["ols_adjusted_r_squared"],
        "Holdout R-squared": summary["holdout_r_squared"],
        "Holdout RMSE": summary["holdout_rmse"],
    }
    st.json(diagnostics)
    st.download_button(
        "Download analysis JSON",
        data=json.dumps(result.to_dict(), indent=2),
        file_name="aspect_sentiment_analysis.json",
        mime="application/json",
    )
