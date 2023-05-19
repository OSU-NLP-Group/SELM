"""
Checks for experiments that have not run.
"""
import argparse

from .. import config, util
from . import lib


def parse_args() -> argparse.Namespace:
    # check for finished experiments
    parser = argparse.ArgumentParser(
        description="Check which experiments still need to run. This will dirty your relics directory in git. You most likely want to make sure your relics directory is clean, then run this command, then run `git clean -f relics`.",
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        type=str,
        help="Config .toml files or directories containing config .toml files.",
    )
    parser.add_argument(
        "--regex",
        action="store_true",
        help="Whether to use regular expression matching on [experiments] argument",
        default=False,
    )

    return parser.parse_args()


def check(args: argparse.Namespace) -> None:
    if args.regex:
        iterator = util.files_with_match(args.experiments)
    else:
        iterator = util.files_with_extension(args.experiments, ".toml")

    for experiment_toml in iterator:
        # If there are any configs that haven't run, print the file name.
        for experiment_config in config.load_configs(experiment_toml):
            experiment = lib.experiment_from_config(experiment_config)
            finished_trials = sum(
                1 for t in experiment if "finished" in t and t["finished"]
            )
            if finished_trials < experiment_config.trials:
                print(experiment_toml)
                break


def main():
    args = parse_args()
    check(args)


if __name__ == "__main__":
    main()
