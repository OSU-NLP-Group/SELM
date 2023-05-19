"""
This module writes some examples with min/max perplexity for a given model.

It also includes some helpers for loading the data for training purposes.
"""

import collections
import dataclasses
import itertools
import typing
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import relic
import sklearn

from .. import logging, relic_helpers
from . import helpers

logger = logging.init(__name__)

PreprocessFn = Callable[[np.ndarray], np.ndarray]

Feature = Literal["l2-norm", "l1-norm", "max", "min", "mean", "std"]

FEATURE_FUNCTIONS: Dict[Feature, PreprocessFn] = {
    "l2-norm": lambda x: np.linalg.norm(x, ord=2, axis=1),
    "l1-norm": lambda x: np.linalg.norm(x, ord=1, axis=1),
    "max": lambda x: np.max(x, axis=1),
    "min": lambda x: np.min(x, axis=1),
    "mean": lambda x: np.mean(x, axis=1),
    "std": lambda x: np.std(x, axis=1),
}


helpers.insist(
    sorted(typing.get_args(Feature)) == sorted(FEATURE_FUNCTIONS.keys()),
    "There is a missing feature function!",
)


def calc_features(x, *, features: Sequence[Feature]) -> np.ndarray:
    n_examples, n_features = x.shape

    assert x.shape == (n_examples, n_features)

    # Hand-crafted features
    processed = np.stack([FEATURE_FUNCTIONS[feature](x) for feature in features]).T

    assert processed.shape == (n_examples, len(features))

    return processed


def preprocess(x: np.ndarray) -> np.ndarray:
    features = list(sorted(FEATURE_FUNCTIONS.keys()))
    # features = ["l2-norm", "std"]
    return calc_features(x, features=features)


def load_experiments(name, fixed_key=False, no_regularization=False):
    filters = [
        f"(~ data.file |adversarial/{name}|)",
        "(== training.regularization.weight 100000)",
    ]

    if fixed_key:
        filters += ["(== seed 0)"]
    else:
        # Either not seed 0, or the ONE experiment that used a seed of 0
        # (when will I learn not to do this?)
        filters += [
            "(or (not (== seed 0)) (== experimenthash 'c8a19f027fea026740765c8ffaaa975335987625'))"
        ]

    if no_regularization:
        filters += ["(== training.regularization.variety None)"]
    else:
        filters += ["(== training.regularization.variety 'target-l2-norm')"]

    exps = list(
        relic.load_experiments(
            filter_fn=relic.cli.lib.shared.make_experiment_fn(filters)
        )
    )
    if not exps:
        raise RuntimeError(f"There are no experiments matching filters {filters}")
    return exps


def load_ciphertexts(
    experiments: List[relic.Experiment], max_count: int
) -> Dict[str, np.ndarray]:
    """
    Ciphertexts should be returned as numpy matrices, grouped by the file.

    Ciphertexts with duplicate keys are a problem because they are effectively duplicate ciphertexts.

    Experiments should only differ by file; any other difference is likely a mistake. (Maybe this can be overridden by a CLI option in the future).
    """
    helpers.insist(experiments, "No experiments!")

    diffs = relic.experiments.differing_config_fields(experiments)
    helpers.insist(diffs == ["data.file"], diffs)

    dim = experiments[0].config["model"]["intrinsic_dimension"]
    helpers.insist(dim, "Need an intrinsic dimension")

    # Lookup from "plaintext file" to List[relic.Experiment]
    exp_lookup = collections.defaultdict(list)
    for exp in experiments:
        exp_lookup[exp.config["data"]["file"]].append(exp)

    ciphertexts = {}
    for file, exps in exp_lookup.items():
        dataset, _ = load_dataset(exps)

        # Make sure to include on max_count ciphertexts
        dataset = dataset[:max_count, :]
        # Assert it because I don't trust myself
        n, dim = dataset.shape
        assert n <= max_count, f"{file} has {n} ciphertexts!"

        ciphertexts[file] = dataset

    return ciphertexts


def load_dataset(exps: List[relic.Experiment]) -> Tuple[np.ndarray, List[str]]:
    """
    Loads all the ciphertexts for a given dataset.
    """

    ciphertexts, hashes = [], []
    for exp in sorted(exps, key=lambda e: e.config["seed"]):
        for t, _ in enumerate(exp):
            if not exp[t]["finished"]:
                logger.info(f"Trial {t} has not finished. Skipping.")
                continue

            if not exp[t]["succeeded"]:
                logger.info(f"Trial {t} failed! Skipping.")
                continue

            if not exp.model_exists(t):
                logger.warning(
                    "Trial succeeded, but ciphertext does not exist! [exp: %s, trial: %s]",
                    exp,
                    t,
                )
                continue

            ciphertexts.append(relic_helpers.load_ciphertext(exp, t))
            hashes.append(exp.hash)

    return np.stack(ciphertexts), hashes


def make_train_test_split(
    arr: np.ndarray, ratio: float
) -> Tuple[np.ndarray, np.ndarray]:
    split = int(len(arr) * ratio)
    train, test = arr[:split], arr[split:]

    return train, test


def join_pair(a, b):
    x = np.concatenate((a, b))
    y = np.zeros(x.shape[0])
    y[a.shape[0] :] = 0
    y[: a.shape[0]] = 1

    return x, y


@dataclasses.dataclass(frozen=True)
class SingleDataset:
    name: str
    splits: Tuple[np.ndarray, ...]


def make_single_datasets(
    ciphertexts: Dict[str, np.ndarray],
    seed: int,
    *,
    preprocess_fn: Optional[PreprocessFn] = None,
    ratio: float = 0.8,
) -> Iterator[SingleDataset]:
    for file, stack in ciphertexts.items():
        stack = sklearn.utils.shuffle(stack, random_state=seed)
        train, test = make_train_test_split(stack, ratio=ratio)
        if preprocess_fn:
            train = preprocess_fn(train)
            test = preprocess_fn(test)
        yield SingleDataset(file, (train, test))


@dataclasses.dataclass(frozen=True)
class PairedDataset:
    name: str
    splits: Tuple[Tuple[np.ndarray, np.ndarray], ...]
    files: Tuple[str, str]


def make_paired_datasets(
    datasets: Iterable[SingleDataset], seed: int, left: Optional[str] = None
) -> Iterator[PairedDataset]:
    for a, b in itertools.combinations(datasets, 2):
        if left:
            if b.name == left:
                a, b = b, a
            elif a.name == left:
                pass
            else:
                continue

        assert len(a.splits) == len(b.splits)

        splits = []

        for split_a, split_b in zip(a.splits, b.splits):
            x, y = join_pair(split_a, split_b)
            x, y = sklearn.utils.shuffle(x, y, random_state=seed)
            splits.append((x, y))

        assert len(splits) == len(a.splits) == len(b.splits)

        yield PairedDataset(f"{a.name} vs. {b.name}", tuple(splits), (a.name, b.name))
