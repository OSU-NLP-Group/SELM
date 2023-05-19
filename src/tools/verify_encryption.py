"""

"""
import argparse
import logging

import relic
import torch
from tqdm.auto import tqdm

log_format = "[%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.WARNING, format=log_format)
logger = logging.getLogger("verify-enc")

from .. import accelerate, config, evaluating, modeling, relic_helpers, tokenizing


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiments",
        nargs="+",
        help="Filter experiments based on results. Example: '(all (< epochs 1000))'",
        default=[],
    )
    parser.add_argument(
        "--relics",
        help="Path to relics/ directory",
    )

    return parser


def check_trial_succeeded(exp: relic.Experiment, trial: int) -> bool:
    if trial > len(exp):
        logger.warning("Trial missing. [exp: %s, trial: %d]", exp.hash, trial)
        return False

    if not exp[trial]["finished"]:
        logger.warning("Trial not finished. [exp: %s, trial: %d]", exp.hash, trial)
        return False

    if not exp[trial]["succeeded"]:
        logger.warning("Trial failed. [exp %s, trial: %d]", exp.hash, trial)
        return False

    if not exp.model_exists(trial):
        logger.warning("Model missing. [exp: %s, trial: %d]", exp.hash, trial)
        return False

    return True


def verify_trial(exp: relic.Experiment, trial: int) -> bool:
    saved = torch.load(exp.model_path(trial))
    seed = saved["fastfood_seed"]
    theta_d = saved["theta_d"]

    experiment_config = config.ExperimentConfig.from_dict(exp.config)

    tokenizer = tokenizing.new(experiment_config.tokenizer)

    model = modeling.new(
        experiment_config.model,
        vocab=len(tokenizer),
        seed=seed,
    )

    accelerate.prepare(model)

    with torch.no_grad():
        model.intrinsic_vector.copy_(theta_d)
        model.set_module_weights()

    return evaluating.passes(model, tokenizer, experiment_config, exp[trial]["epochs"])


def main():
    parser = init_parser()
    args = parser.parse_args()

    experiments = relic_helpers.load_experiments(args.experiments, args.relics)

    for exp in experiments:
        for trial, _ in enumerate(tqdm(exp)):
            if not check_trial_succeeded(exp, trial):
                continue

            if not verify_trial(exp, trial):
                print(f"{exp.hash[:8]} {trial}")


if __name__ == "__main__":
    main()
