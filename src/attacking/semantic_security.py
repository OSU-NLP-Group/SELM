import dataclasses
from typing import Any, Callable, Dict, Iterator, Optional, Protocol

import numpy as np
import scipy.stats

from .. import logging
from . import data

logger = logging.init(__name__, date=False, verbose=True)


class Model(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


TrainingCallback = Callable[[Model, data.PairedDataset], None]


@dataclasses.dataclass
class SemanticSecurityConfig:
    plaintext_a: str
    plaintext_b: str
    params: Dict[str, Any]


def confidence_interval(n_test, confidence=0.95):
    percentages = (1 - confidence) / 2, confidence + (1 - confidence) / 2
    logger.debug(
        "Calculated percentages. [confidence: %.3g, lower: %.3g, upper: %.3g]",
        confidence,
        *percentages,
    )
    # Coin flip
    p = 0.5
    # Binomial distribution
    lower, upper = scipy.stats.binom.ppf(percentages, n_test, p) / n_test
    return lower, upper


def play(
    datasets: Iterator[data.SingleDataset],
    model_fn: Callable[[], Model],
    model_name: str,
    seed: int,
    trained_model_callback: Optional[TrainingCallback] = None,
    quiet: bool = False,
) -> bool:
    logger = logging.init(model_name)
    passed = True

    for pair in data.make_paired_datasets(
        datasets, seed, left="data/news/100-tokens/0.txt"
    ):
        model = model_fn()

        (train_x, train_y), (test_x, test_y) = pair.splits

        if not quiet:
            logger.info("Starting model.fit")

        model.fit(train_x, train_y)

        if callable(trained_model_callback):
            trained_model_callback(model, pair)

        train_score = model.score(train_x, train_y)
        test_score = model.score(test_x, test_y)

        params = getattr(model, "best_params_", {})
        if not quiet:
            logger.info(
                "Fitted. [pair: %s, train acc: %.2f, test acc: %.2f, params: %s]",
                pair.name,
                train_score,
                test_score,
                params,
            )

        n_test = len(test_y)
        lower, upper = confidence_interval(n_test)

        n_correct = 0
        for label, prediction in zip(test_y, model.predict(test_x)):
            if label == prediction:
                n_correct += 1

        test = scipy.stats.binomtest(n_correct, n_test, p=0.5, alternative="greater")

        if test.pvalue < 0.05:  # if failed
            logger.warn(
                "Reject null. [pair: %s, test acc: %.3f, p: %.3g]",
                pair.name,
                test_score,
                test.pvalue,
            )
            passed = False

        if test_score > upper and not quiet:
            logger.warn(
                "Outside confidence interval. [pair: %s, test acc: %.3f, upper: %.3f]",
                pair.name,
                test_score,
                upper,
            )
            passed = False

        if test.pvalue > 0.05 and test_score < upper:
            logger.info(
                "Fail to reject. [pair: %s, test acc: %.3f, upper: %.3f, p: %.3g]",
                pair.name,
                test_score,
                upper,
                test.pvalue,
            )

    return passed
