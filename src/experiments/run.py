import argparse
import os
import secrets

import preface

from .. import config, logging, modeling, modeling_utils, tokenizing, training
from . import lib


def run(experiment_config: config.ExperimentConfig) -> None:
    experiment = lib.experiment_from_config(experiment_config)

    for trial_num in range(experiment_config.trials):
        # Check if we need to run a trial at all.
        # TODO: Refactor into a function.
        if trial_num < len(experiment):
            if (
                "finished" not in experiment[trial_num]
                or not experiment[trial_num]["finished"]
            ):
                logger.info(
                    "Trial is not finished. We can't continue training from a partially-trained model, so we are restarting this trial. [experiment %s, trial: %s]",
                    experiment,
                    trial_num,
                )
                experiment.delete_trials(trial_num)
            else:
                logger.info("Trial is already finished. [trial: %s]", trial_num)
                if not experiment_config.save_weights:
                    logger.info(
                        "Experiment does not need any saved weights. Skipping trial. [trial: %s]",
                        trial_num,
                    )
                    continue

                if experiment.model_exists(trial_num):
                    logger.info(
                        "Experiment needs saved weights and saved weights found. Skipping trial. [experiment: %s, trial: %s, weights: %s]",
                        experiment,
                        trial_num,
                        experiment.model_path(trial_num),
                    )
                    continue

                logger.warn(
                    "Experiment needs saved weights but trial has no weights! Removing this trial and all future trials and starting again. [experiment: %s, trial: %s, weights: %s]",
                    experiment,
                    trial_num,
                    experiment.model_path(trial_num),
                )
                experiment.delete_trials(trial_num)

        # If we get here, then we need to run a trial.
        logger.info(
            "Starting trial. [trial: %s, config: %s]", trial_num, experiment_config
        )

        tokenizer = tokenizing.new(experiment_config)
        dataset = tokenizing.make_dataset(experiment_config.data, tokenizer)

        if experiment_config.seed_source == "config":
            seed = experiment_config.seed
        elif experiment_config.seed_source == "trial":
            seed = trial_num
        elif experiment_config.seed_source == "random":
            seed = secrets.randbits(32)
        else:
            preface.never(experiment_config.seed_source)

        model = modeling.new(
            experiment_config.model,
            vocab=len(tokenizer),
            seed=seed,
        )

        if experiment_config.save_weights:
            assert isinstance(model, modeling_utils.Saveable)

        logger.info("Loaded model and tokenizer.")

        trial, model = training.train(
            model, dataset, tokenizer, experiment_config, experiment
        )

        logger.info("Finished training!")

        model_path = None
        if experiment_config.save_weights:
            model_path = f"{experiment.hash}-{trial_num}-temporary.bin"
            model.save(model_path)

        experiment.update_trial(trial, model_path)
        if model_path:
            os.remove(model_path)

        logger.info("Finished trial. [trial: %s]", trial_num)


def run_all(args: argparse.Namespace) -> None:
    for experiment_config in lib.find_experiments(args.experiments):
        run(experiment_config)


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument(
        "--verbose",
        help="Whether to provide verbose output.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "experiments",
        nargs="+",
        type=str,
        help="Paths to directories containing config.toml files OR just a config.toml file. Directories will be searched for any nested config.toml files.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger = logging.init("experiments.run", args.verbose)

    run_all(args)
