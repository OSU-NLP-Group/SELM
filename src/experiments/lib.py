import dataclasses
import logging
import pathlib
from typing import Any, Dict, Iterator

import relic

from .. import config, util

logger = logging.getLogger(__name__)


def find_experiments(paths) -> Iterator[config.ExperimentConfig]:
    """
    Arguments:
    * args (list[str]): list of strings that are either directories containing files or config files themselves.
    """
    if not isinstance(paths, list):
        paths = [paths]

    for config_file in util.files_with_extension(paths, ".toml"):
        yield from config.load_configs(config_file)


def make_relic_config(experiment_config: config.ExperimentConfig) -> Dict[str, Any]:
    relic_config = dataclasses.asdict(experiment_config)

    # don't want to include these parameters in the relic repository.
    relic_config.pop("trials")
    relic_config.pop("save_weights")
    relic_config["training"].pop("maximum_epochs")
    relic_config["training"].pop("snapshot_interval")
    relic_config["training"].pop("report_interval")

    return relic_config


def experiment_from_config(
    experiment_config: config.ExperimentConfig,
) -> relic.Experiment:
    """
    Create a relic experiment from an ExperimentConfig. This method removes some fields from ExperimentConfig that shouldn't matter when considering results (whether the model was saved, how many trials were run, etc.).
    """
    relic_exp = relic.new_experiment(
        make_relic_config(experiment_config), pathlib.Path("relics")
    )

    return relic_exp
