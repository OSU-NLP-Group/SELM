import argparse
import pathlib
import shutil
import threading
from typing import List, Optional, Sequence, Set

import relic
import torch

from . import logging


def load_ciphertext(exp, trial, as_tensor=False):
    assert exp.model_exists(trial), f"{exp.hash}, {trial}"
    saved = torch.load(exp.model_path(trial))
    ciphertext = saved["theta_d"]
    if as_tensor:
        return ciphertext

    return ciphertext.numpy()


def load_experiments(
    filters: Sequence[str], *, root=None, show_cmd=False
) -> List[relic.Experiment]:
    kwargs = {}
    if root is not None:
        kwargs["root"] = pathlib.Path(root)

    if show_cmd:
        filter_str = " ".join([f'"{f}"' for f in filters])
        cmd = "relic ls"
        if root is not None:
            cmd += f" --root {root}"
        cmd += f" --experiments {filter_str}"
        print(cmd)

    filter_fn, needs_trials = relic.cli.lib.shared.make_experiment_fn(filters)

    return list(
        relic.load_experiments(filter_fn=filter_fn, needs_trials=needs_trials, **kwargs)
    )


class TrialManager:
    _trial: Optional[relic.Trial]

    def __init__(self):
        self._trial = None

    def register_trial(self, trial: relic.Trial, experiment: relic.Experiment) -> None:
        self._trial = trial
        self._experiment = experiment

    def check_trial(self) -> None:
        if self._trial is None:
            raise RuntimeError("Need to register a trial before logging anything")

    def log_step(self, **kwargs) -> None:
        _acquireLock()
        try:
            self.check_trial()

            for key, value in kwargs.items():
                if key not in self._trial:
                    self._trial[key] = []

                self._trial[key].append(value)

        finally:
            _releaseLock()

    def log_metric(self, **kwargs) -> None:
        _acquireLock()
        try:
            self.check_trial()
            self._trial.update(kwargs)
        finally:
            _releaseLock()

    @property
    def trial(self) -> relic.Trial:
        return self._trial

    def commit(self) -> None:
        _acquireLock()
        try:
            self._experiment.update_trial(self._trial)
        finally:
            _releaseLock()


_manager = TrialManager()
_enabled = True


def commit() -> None:
    if _enabled:
        _manager.commit()


def register_trial(trial: relic.Trial, experiment: relic.Experiment) -> None:
    if _enabled:
        _manager.register_trial(trial, experiment)


def log_step(**kwargs) -> None:
    if _enabled:
        _manager.log_step(**kwargs)


def log_metric(**kwargs) -> None:
    if _enabled:
        _manager.log_metric(**kwargs)

def disable() -> None:
    global _enabled
    _enabled = False



# From https://github.com/python/cpython/blob/3.10/Lib/logging/__init__.py
_lock = threading.RLock()


def _acquireLock() -> None:
    """
    Acquire the module-level lock for serializing access to shared data.
    This should be released with _releaseLock().
    """
    if _lock:
        _lock.acquire()


def _releaseLock() -> None:
    """
    Release the module-level lock acquired by calling _acquireLock().
    """
    if _lock:
        _lock.release()


def delete_trials(experiment: relic.Experiment, bad_trials: Set[int]) -> None:
    # Need to delete the bad trials
    for i in reversed(sorted(bad_trials)):
        experiment._delete_model(i)
        experiment.trial_path(experiment.root, experiment.hash, i).unlink()
        del experiment.trials[i]

    # Then move all other trials *down* an instance
    for i, trial in enumerate(experiment):
        if trial.instance != i:
            if experiment.model_exists(trial.instance):
                shutil.copy(
                    experiment.model_path(trial.instance), experiment.model_path(i)
                )
            trial["instance"] = i

    # Remove the trial directory so there are no bad trials left behind
    shutil.rmtree(experiment.trial_dir(experiment.root, experiment.hash))

    # Save the experiment again
    experiment.save()


def clean_unfinished_trials(filters: Sequence[str]) -> None:
    """
    Deletes any trials that are unfinished.
    """

    logger = logging.init("clean-unfinished")

    for experiment in load_experiments(filters):
        to_delete = set()
        for i, trial in enumerate(experiment):
            assert trial.instance == i
            if "finished" in trial and trial["finished"]:
                continue

            to_delete.add(i)

        if not to_delete:
            logger.info("All trials finished. [hash: %s]", experiment.hash)
            continue

        logger.info(
            "Found unfinished trials. [hash: %s, count: %d]",
            experiment.hash,
            len(to_delete),
        )

        delete_trials(experiment, to_delete)


def clean_duplicate_seed_trials(filters: List[str]) -> None:
    logger = logging.init("clean-duplicate-seed")

    for experiment in load_experiments(filters + ["(== seed_source 'random')"]):
        to_delete = set()
        seen_seeds = set()
        for i, trial in enumerate(experiment):
            assert trial.instance == i
            if not experiment.model_exists(i):
                continue

            seed = torch.load(experiment.model_path(i))["fastfood_seed"]

            if seed in seen_seeds:
                to_delete.add(i)
            else:
                seen_seeds.add(seed)

        if not to_delete:
            logger.info("All trials have unique seeds. [hash: %s]", experiment.hash)
            continue

        logger.info(
            "Found duplicate seeds. [hash: %s, count: %d]",
            experiment.hash,
            len(to_delete),
        )

        delete_trials(experiment, to_delete)


def clean_bad_seed_trials(filters: Sequence[str]) -> None:
    logger = logging.init("clean-bad-seed")

    for exp in load_experiments(filters):
        if exp.config["seed_source"] == "random":
            continue
        to_delete = set()

        for i, trial in enumerate(exp):
            assert trial.instance == i
            if not exp.model_exists(i):
                continue

            seed = torch.load(exp.model_path(i))["fastfood_seed"]
            if exp.config["seed_source"] == "trial":
                if seed != i:
                    to_delete.add(i)
            elif exp.config["seed_source"] == "config":
                if seed != exp.config["seed"]:
                    to_delete.add(i)

            else:
                raise ValueError(exp.config["seed_source"])

        if not to_delete:
            logger.info("All trials have good seeds. [hash: %s]", exp.hash)
            continue

        logger.info("Found bad seeds. [hash: %s, count: %d]", exp.hash, len(to_delete))

        delete_trials(exp, to_delete)


def clean_missing_model_trials(filters: Sequence[str]) -> None:
    for exp in load_experiments(filters):
        to_delete = []

        for i, trial in enumerate(exp):
            assert trial.instance == i
            if exp.model_exists(i):
                continue

            if not trial["succeeded"]:
                continue

            # Now the trial succeeded, but the model doesn't exist
            to_delete.append(i)

        if not to_delete:
            continue

        assert sorted(to_delete) == to_delete

        delete_trials(exp, set(to_delete))


def clean_tensors(filters: Sequence[str], root=None) -> None:
    logger = logging.init("clean-tensors")

    for exp in load_experiments(filters, root=root):
        for i, trial in enumerate(exp):
            assert trial.instance == i

            for key, value in trial.items():
                if not isinstance(value, list):
                    continue

                try:
                    trial[key] = torch.tensor(value).tolist()
                except TypeError:
                    logger.warn(key)

            exp.update_trial(trial)


def clean(args: argparse.Namespace) -> None:
    if args.duplicate_seed:
        clean_duplicate_seed_trials(args.experiments)

    if args.bad_seed:
        clean_bad_seed_trials(args.experiments)

    if args.missing_model:
        clean_missing_model_trials(args.experiments)

    if args.unfinished:
        clean_unfinished_trials(args.experiments)

    if args.tensors:
        clean_tensors(args.experiments, args.root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Available commands.")

    clean_parser = subparsers.add_parser(
        "clean", help="Remove different experiments that have an issue of some kind."
    )
    clean_parser.add_argument(
        "--unfinished", action="store_true", help="Remove all unfinished trials"
    )
    clean_parser.add_argument(
        "--duplicate-seed",
        action="store_true",
        help="Remove all trials with duplicate random seeds",
    )
    clean_parser.add_argument(
        "--bad-seed",
        action="store_true",
        help="Remove all trials where the seed doesn't match the config.",
    )
    clean_parser.add_argument(
        "--missing-model",
        action="store_true",
        help="Remove all trials with a missing model even if they succeeded",
    )
    clean_parser.add_argument(
        "--tensors",
        action="store_true",
        help="Change all lists of single-item tensors to a multi-item tensor",
    )
    clean_parser.add_argument(
        "--root",
        help="Project root",
    )
    clean_parser.set_defaults(fn=clean)
    relic.cli.lib.shared.add_filter_options(clean_parser)

    args = parser.parse_args()
    if hasattr(args, "fn"):
        args.fn(args)
    else:
        parser.print_help()
