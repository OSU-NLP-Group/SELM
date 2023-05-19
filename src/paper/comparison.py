"""
Script to parse the .stats file and check how different ciphers perform.

For each cipher, we want speed in bytes/second (or similar), as well as the ciphertext size ratio.
"""
import argparse
import dataclasses
import statistics
from typing import Iterator, List, Tuple

import matplotlib.pyplot as plt
import relic

PLAINTEXT_KEY = "plaintext-size: "
CIPHERTEXT_KEY = "ciphertext-size: "
TIME_KEY = "real: "


@dataclasses.dataclass(frozen=True)
class Stats:
    key: str
    time: float
    plaintext: int
    ciphertext: int

    @property
    def size(self):
        """
        Ciphertext size as a ratio of the original plaintext
        """
        return float(self.ciphertext) / float(self.plaintext)

    @property
    def speed(self):
        """
        Speed in bytes/second.
        """
        return self.plaintext / self.time

    @classmethod
    def mean_of_many(cls, stats: List["Stats"]) -> "Stats":
        assert all(stat.key == stats[0].key for stat in stats)

        mean_time = statistics.mean(stat.time for stat in stats)
        mean_plaintext = int(statistics.mean(stat.plaintext for stat in stats))
        mean_ciphertext = int(statistics.mean(stat.ciphertext for stat in stats))

        return cls(stats[0].key, mean_time, mean_plaintext, mean_ciphertext)


def parse_algorithm(lines, i) -> Tuple[Stats, int]:
    while not lines[i]:
        i += 1
    key = lines[i]

    times = []

    i += 1
    while lines[i].startswith(TIME_KEY):
        times.append(float(lines[i][len(TIME_KEY) :]))
        i += 1

    assert lines[i].startswith(PLAINTEXT_KEY)
    plaintext = int(lines[i][len(PLAINTEXT_KEY) :])
    i += 1

    assert lines[i].startswith(CIPHERTEXT_KEY)
    ciphertext = int(lines[i][len(CIPHERTEXT_KEY) :])
    i += 2

    return Stats(key, statistics.mean(times), plaintext, ciphertext), i


def parse_file(filepath: str) -> List[Stats]:
    with open(filepath) as file:
        lines = [line.strip() for line in file]

    i = 0
    stats = []
    while i < len(lines):
        stat, i = parse_algorithm(lines, i)
        stats.append(stat)
    return stats


def _calc_ciphertext_size(exp: relic.Experiment) -> int:
    return exp.config["model"]["intrinsic_dimension"] * 4 + 4


def _calc_plaintext_size(exp: relic.Experiment) -> int:
    with open(exp.config["data"]["file"], "rb") as file:
        return len(file.read())


def find_experiments(model: str, size: int) -> Iterator[Stats]:
    filter_fn, needs_trials = relic.cli.lib.shared.make_experiment_fn(
        [
            f"(== model.intrinsic_dimension {size})",
            f"(== model.language_model_name_or_path '{model}')",
            "(not (or (~ data.file 'random') (~ data.file 'aesw')))",
        ]
    )
    for experiment in relic.load_experiments(
        filter_fn=filter_fn, needs_trials=needs_trials
    ):
        ciphertext_size = _calc_ciphertext_size(experiment)
        plaintext_size = _calc_plaintext_size(experiment)
        seconds = []
        for trial in experiment:
            if (
                "finished" not in trial
                or not trial["finished"]
                or "succeeded" not in trial
                or not trial["succeeded"]
            ):
                continue

            if trial["epochs"] <= 100:
                continue

            seconds.append(trial["seconds_per_epoch"] * trial["epochs"])

        if not seconds:
            continue

        yield Stats(
            f"{model}-{size}", statistics.mean(seconds), plaintext_size, ciphertext_size
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help=".stats file to parse")

    args = parser.parse_args()

    traditional_stats = parse_file(args.file)

    for stat in traditional_stats:
        print(f"{stat.key}\t{stat.speed:.2g}\t{stat.size:.2g}")

    return

    gpt2_10K = Stats.mean_of_many(list(find_experiments("gpt2", 10000)))
    gpt2_100K = Stats.mean_of_many(list(find_experiments("gpt2", 100000)))

    fig, ax = plt.subplots(1, 1)
    ax.set_yscale("log")
    ax.set_xscale("log")

    for statgroup in [traditional_stats]:  # , [gpt2_10K], [gpt2_100K]]:
        x = [stat.speed for stat in statgroup]
        y = [stat.size for stat in statgroup]

        ax.scatter(x, y)

    plt.show()


if __name__ == "__main__":
    main()
