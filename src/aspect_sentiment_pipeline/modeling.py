from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ModelConfig:
    min_mentions: int = 3
    variance_threshold: float = 0.85
    max_components: int = 12
    test_size: float = 0.25
    random_state: int = 42

    def validate(self) -> None:
        if self.min_mentions < 1:
            raise ValueError("min_mentions must be at least 1.")
        if not 0.5 <= self.variance_threshold <= 1.0:
            raise ValueError("variance_threshold must be between 0.5 and 1.0.")
        if self.max_components < 1:
            raise ValueError("max_components must be at least 1.")
        if not 0.1 <= self.test_size <= 0.5:
            raise ValueError("test_size must be between 0.1 and 0.5.")


@dataclass(frozen=True)
class RatingDriverModel:
    aspect_statistics: pd.DataFrame
    retained_aspects: list[str]
    component_count: int
    explained_variance: float
    intercept: float
    r_squared: float
    adjusted_r_squared: float
    holdout_r_squared: float | None
    holdout_mae: float
    holdout_rmse: float


def fit_rating_driver_model(
    aspect_values: pd.DataFrame,
    aspect_mentioned: pd.DataFrame,
    ratings: pd.Series,
    config: ModelConfig,
) -> RatingDriverModel:
    config.validate()
    mention_counts = aspect_mentioned.sum(axis=0).astype(int)
    retained = mention_counts[mention_counts >= config.min_mentions].index.tolist()
    if not retained:
        raise ValueError(
            "No aspect meets the minimum mention count. Add more reviews or lower --min-mentions."
        )

    x = aspect_values[retained].to_numpy(dtype=float)
    y = ratings.to_numpy(dtype=float)
    if np.allclose(x, 0):
        raise ValueError("All extracted aspect sentiment scores are neutral.")

    reduced, components, component_count, explained_variance = _reduce_features(x, config)
    design = sm.add_constant(reduced, has_constant="add")
    fitted = sm.OLS(y, design).fit(cov_type="HC3")

    latent_coefficients = np.asarray(fitted.params[1:], dtype=float)
    aspect_coefficients = components.T @ latent_coefficients
    latent_covariance = np.asarray(fitted.cov_params(), dtype=float)[1:, 1:]
    degrees_of_freedom = max(1, int(fitted.df_resid))
    critical_value = float(student_t.ppf(0.975, degrees_of_freedom))

    statistics = []
    for aspect_index, aspect in enumerate(retained):
        loading = components[:, aspect_index]
        coefficient = float(aspect_coefficients[aspect_index])
        variance = float(loading @ latent_covariance @ loading.T)
        standard_error = sqrt(max(variance, 0.0))
        if abs(coefficient) < 1e-10 and standard_error < 1e-10:
            coefficient = 0.0
            standard_error = 0.0
        test_statistic = coefficient / standard_error if standard_error else 0.0
        p_value = (
            float(2 * student_t.sf(abs(test_statistic), degrees_of_freedom))
            if standard_error
            else 1.0
        )
        count = int(mention_counts[aspect])
        coverage = count / len(aspect_values)
        mean_sentiment = float(aspect_values.loc[aspect_mentioned[aspect], aspect].mean())
        statistics.append(
            {
                "aspect": aspect,
                "mentions": count,
                "coverage": coverage,
                "mean_sentiment": mean_sentiment,
                "rating_impact": coefficient,
                "standard_error": standard_error,
                "p_value": p_value,
                "confidence_low": coefficient - critical_value * standard_error,
                "confidence_high": coefficient + critical_value * standard_error,
                "importance_score": abs(coefficient) * sqrt(coverage),
                "current_rating_effect": coefficient * mean_sentiment,
            }
        )

    aspect_statistics = pd.DataFrame(statistics).sort_values(
        "importance_score", ascending=False, ignore_index=True
    )
    holdout = _evaluate_holdout(x, y, component_count, config)

    return RatingDriverModel(
        aspect_statistics=aspect_statistics,
        retained_aspects=retained,
        component_count=component_count,
        explained_variance=explained_variance,
        intercept=float(fitted.params[0]),
        r_squared=float(fitted.rsquared),
        adjusted_r_squared=float(fitted.rsquared_adj),
        holdout_r_squared=holdout[0],
        holdout_mae=holdout[1],
        holdout_rmse=holdout[2],
    )


def _reduce_features(
    x: np.ndarray,
    config: ModelConfig,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    sample_count, feature_count = x.shape
    if feature_count == 1:
        return x.copy(), np.ones((1, 1)), 1, 1.0

    maximum = max(1, min(config.max_components, feature_count - 1, sample_count - 1))
    probe = TruncatedSVD(n_components=maximum, random_state=config.random_state)
    probe.fit(x)
    cumulative = np.cumsum(probe.explained_variance_ratio_)
    threshold_matches = np.flatnonzero(cumulative >= config.variance_threshold)
    component_count = int(threshold_matches[0] + 1) if len(threshold_matches) else maximum

    reducer = TruncatedSVD(n_components=component_count, random_state=config.random_state)
    reduced = reducer.fit_transform(x)
    explained_variance = float(reducer.explained_variance_ratio_.sum())
    return reduced, reducer.components_, component_count, explained_variance


def _evaluate_holdout(
    x: np.ndarray,
    y: np.ndarray,
    component_count: int,
    config: ModelConfig,
) -> tuple[float | None, float, float]:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    if x.shape[1] == 1:
        train_reduced = x_train
        test_reduced = x_test
    else:
        evaluation_components = max(
            1,
            min(component_count, x.shape[1] - 1, len(x_train) - 1),
        )
        reducer = TruncatedSVD(
            n_components=evaluation_components,
            random_state=config.random_state,
        )
        train_reduced = reducer.fit_transform(x_train)
        test_reduced = reducer.transform(x_test)

    predictor = LinearRegression().fit(train_reduced, y_train)
    predictions = predictor.predict(test_reduced)
    r_squared = float(r2_score(y_test, predictions)) if len(y_test) >= 2 else None
    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(mean_squared_error(y_test, predictions) ** 0.5)
    return r_squared, mae, rmse
