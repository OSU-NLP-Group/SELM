"""
Makes the tables for the paper from the experiments.
"""
import argparse
import dataclasses
import re
import statistics
from typing import List, Set

import relic

from .. import relic_helpers


@dataclasses.dataclass(frozen=True)
class MissingTrialDescription:
    exphash: str
    file: str
    regularized: bool
    trial: int
    explanation: str

    def __str__(self) -> str:
        pieces = [
            f"file: {self.file}",
            f"regularized: {self.regularized}",
            f"trial: {self.trial}",
            f"{self.explanation}",
        ]
        if self.exphash:
            pieces.append(f"hash: {self.exphash}")

        return "\t".join(pieces)


@dataclasses.dataclass(frozen=True)
class Statistics:
    mean: float
    std: float

    def __str__(self) -> str:
        return f"${self.mean:.1f} \\pm {self.std:.1f}$"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "category",
        type=str,
        choices=["wikipedia", "pubmed", "images", "random-words", "random-bytes"],
        help="Category",
    )
    parser.add_argument(
        "length",
        type=int,
        choices=[100, 500, 996],
        help="Length in tokens",
    )
    parser.add_argument(
        "--regularized",
        action="store_true",
        help="Whether to look for regularized trials.",
    )

    return parser


def get_experiments(
    category: str, length: int, regularized: bool
) -> List[relic.Experiment]:
    file_filter = f"(~ data.file |{category}/{length}-tokens|)"
    if regularized:
        reg_filter = "(== training.regularization.weight 4000000000)"
    else:
        reg_filter = "(== training.regularization.variety None)"

    filters = [file_filter, reg_filter]

    experiments = relic_helpers.load_experiments(filters)

    assert experiments, "No experiments found!"

    different_fields = set(relic.experiments.differing_config_fields(experiments))

    if not different_fields:
        print("Only one experiment ran.")
    else:
        assert different_fields == {"data.file"}, different_fields

    return experiments


def parse_file_name(file: str) -> str:
    pattern = r"data\/[\w-]+\/\d+-tokens\/(.*?)\.txt"

    m = re.search(pattern, file)
    assert m, file

    return m.group(1)


def make_filename(category: str, length: int, i: int) -> str:
    return f"data/{category}/{length}-tokens/{i}.txt"


def get_missing_trials(
    category: str, length: int, experiments: List[relic.Experiment]
) -> Set[MissingTrialDescription]:
    missing = set()

    unseen_files = set(range(10))
    seen_files = set()

    for exp in sorted(experiments, key=lambda e: e.config["data"]["file"]):
        regularized = exp.config["training"]["regularization"]["variety"] is not None
        file = exp.config["data"]["file"]

        filename = parse_file_name(file)
        seen_files.add(filename)

        if category != "images":
            unseen_files.remove(int(filename))

        for t in range(len(exp), 3):
            missing.add(
                MissingTrialDescription(
                    exp.hash,
                    file,
                    regularized,
                    t,
                    f"Didn't even start trial {t}!",
                )
            )

        if len(exp) < 3:
            continue

        for t, trial in enumerate(exp):
            if not trial["finished"]:
                missing.add(
                    MissingTrialDescription(
                        exp.hash,
                        file,
                        regularized,
                        t,
                        f"Didn't finish trial {t}!",
                    )
                )
                continue

    if category != "images":
        for i in unseen_files:
            for t in range(3):
                missing.add(
                    MissingTrialDescription(
                        "",
                        make_filename(category, length, i),
                        regularized,
                        t,
                        "Didn't start this experiment!",
                    )
                )
    elif len(seen_files) != 10:
        print("Missing files! But I don't know what they are!")
        breakpoint()

    return missing


def calc_statistics(experiments: List[relic.Experiment]) -> Statistics:
    """
    We have 10 experiments each with at least 3 trials.
    We are going to calculate the mean epochs required and the standard deviation.
    """
    assert len(experiments) == 10
    assert all(len(exp) >= 3 for exp in experiments)

    epochs = []
    for exp in experiments:
        for t, trial in enumerate(exp[:3]):
            if not trial["succeeded"]:
                breakpoint
                continue

            epochs.append(trial["epochs"])

    return Statistics(mean=statistics.mean(epochs), std=statistics.stdev(epochs))


def main() -> None:
    parser = make_parser()

    args = parser.parse_args()

    experiments = get_experiments(args.category, args.length, args.regularized)

    missing_trials = get_missing_trials(args.category, args.length, experiments)

    if missing_trials:
        for missing in missing_trials:
            print(missing)
        return

    stats = calc_statistics(experiments)

    print(stats)


if __name__ == "__main__":
    main()
