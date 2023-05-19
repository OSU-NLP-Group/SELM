"""
Script to encrypt a file using GPT-2 (124M)
"""
import argparse
import datetime
import logging

import preface
import torch
import relic

from src import accelerate, config, evaluating, modeling, tokenizing, training

logger = logging.getLogger("SELM")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="+", help="File(s) to encrypt")
    parser.add_argument(
        "--key",
        required=True,
        type=int,
        help="Symmetric key to use as a seed. You can generate a key with:\n\n\tpython -c 'import secrets; print(secrets.randbits(32))'",
    )
    parser.add_argument(
        "--int-dim", type=int, help="Intrinsic dimension.", default=10_000
    )

    return parser.parse_args()


def new_cfg(int_dim, file):
    return config.ExperimentConfig(
        model=config.ModelConfig(
            "gpt2",
            int_dim,
            dropout=0.0,
            int_dim_dropout=0.0,
        ),
        tokenizer="pretrained",
        data=config.DataConfig(file),
        training=config.TrainingConfig(
            maximum_epochs=10_000,
            learning_rate=2e-8,
            report_interval=10,
            lr_scheduler_type="linear-const",
            warmup_epochs=0,
            decay_epochs=2000,
            decayed_proportion=0.1,
            clipping=config.ClippingConfig(
                algorithm="norm",
                value=1e5,
            ),
            regularization=config.RegularizationConfig(
                variety="distribution-difference-integral",
                weight=5e8,
                mean=0,
                std=4e-7,
                schedule="linear",
                warmup=500,
            ),
        ),
        trials=1,
        save_weights=True,
        seed_source="config",
        seed=0,
    )


if __name__ == "__main__":
    args = parse_args()

    for file in args.file:
        exp_cfg = new_cfg(args.int_dim, file)
        tokenizer = tokenizing.new(exp_cfg)
        dataset = tokenizing.make_dataset(exp_cfg.data, tokenizer)
        model = modeling.new(exp_cfg.model, vocab=len(tokenizer), seed=args.key)

        _, model = training.train(model, dataset, tokenizer, exp_cfg)

        model.save(f"{file}.enc")
