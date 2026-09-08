from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .pipeline import AnalysisConfig, analyse_reviews


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract aspect sentiment and estimate which aspects drive review ratings."
    )
    parser.add_argument("input_csv", type=Path, help="CSV file containing review text and ratings")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/latest"))
    parser.add_argument("--text-column", default="review_text")
    parser.add_argument("--rating-column", default="rating")
    parser.add_argument("--aspects-file", type=Path)
    parser.add_argument("--min-mentions", type=int, default=3)
    parser.add_argument("--variance-threshold", type=float, default=0.85)
    parser.add_argument("--max-components", type=int, default=12)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--top-review-count", type=int, default=5)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    frame = pd.read_csv(arguments.input_csv)
    config = AnalysisConfig(
        text_column=arguments.text_column,
        rating_column=arguments.rating_column,
        aspects_file=arguments.aspects_file,
        min_mentions=arguments.min_mentions,
        variance_threshold=arguments.variance_threshold,
        max_components=arguments.max_components,
        test_size=arguments.test_size,
        top_review_count=arguments.top_review_count,
    )
    result = analyse_reviews(frame, config)
    destination = result.save(arguments.output_dir)
    print(f"Analysed {result.summary['analysed_rows']} reviews.")
    print(f"Average rating: {result.summary['average_rating']:.2f}")
    print(f"Results written to: {destination.resolve()}")


if __name__ == "__main__":
    main()
