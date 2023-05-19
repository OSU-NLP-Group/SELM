"""
Measures SELM's performance in speed (bytes/sec) and size (how much larger than the original it is).
"""
import argparse
import os
from typing import List

import pandas as pd
import relic

from .. import relic_helpers, tokenizing
from . import helpers

filters = (
    "(== training.regularization.variety None)",
    "(== training.lr_scheduler_type 'linear-const')",
    "(== training.plateau_reduce_factor 0.1)",
    "(== tokenizer.pretrained 'gpt2')",
    # Fix model size and domain
    "(== model.language_model_name_or_path 'gpt2')",
    "(~ data.file 'news')",
    # Choose either 100 or 1000 token files.
    "(or (~ data.file '100-tokens') (~ data.file '1000-tokens'))",
    # Handle different learning rate for 100K
    r"""(or
            (and
                (< model.intrinsic_dimension 100000)
                (== training.batch_size 1)
                (== training.learning_rate 0.00000002))
            (and
                (== model.intrinsic_dimension 100000)
                (== training.batch_size 2)
                (== training.learning_rate 0.00000001)))""",
)


def make_dataframe(experiments: List[relic.Experiment]) -> pd.DataFrame:
    headers = [
        "tokens",
        "epochs",
        "dimension",
        "input_size",
        "output_size",
        "size_ratio",
        "seconds",
        "speed",
    ]

    rows = []
    for exp in experiments:
        tokens = helpers.parse_length(exp.config["data"]["file"])
        input_size = os.path.getsize(exp.config["data"]["file"])
        dimension = exp.config["model"]["intrinsic_dimension"]
        size = helpers.translate_size(
            exp.config["model"]["language_model_name_or_path"]
        )

        # We just use the default prompt because we are only doing UUID experiments.
        # dimension * 4 for 4 bytes/float, len(tokenizing.DEFAULT_PROMPT) for the
        # shared public prompt, and 4 bytes for x (turning a deterministic cipher
        # into a probabilistic cipher).
        output_size = dimension * 4 + len(str(tokenizing.DEFAULT_PROMPT)) + 4

        if not len(exp):
            continue

        if len(exp) > 1:
            print(exp.hash, len(exp))

        trial = exp[0]
        if not trial["finished"] or not trial["succeeded"]:
            continue

        epochs = trial["epochs"]
        seconds = trial["seconds_per_epoch"] * epochs

        rows.append(
            [
                tokens,
                epochs,
                dimension,
                input_size,
                output_size,
                output_size / input_size,
                seconds,
                input_size / seconds,
            ]
        )

    return pd.DataFrame(data=rows, columns=headers)


def print_table(df):
    def print_row(rows):
        print(
            "tokens:",
            rows.tokens.iloc[0],
            "dimension:",
            rows.dimension.iloc[0],
            "examples:",
            len(rows),
        )
        rows = rows[["speed", "size_ratio"]]
        print(rows.mean(numeric_only=True))
        print(rows.std(numeric_only=True))

    print_row(df[(df["tokens"] == 100) & (df["dimension"] == 1000)])
    print_row(df[(df["tokens"] == 100) & (df["dimension"] == 10000)])
    print_row(df[(df["tokens"] == 100) & (df["dimension"] == 100000)])
    print_row(df[(df["tokens"] == 1000) & (df["dimension"] == 10000)])
    print_row(df[(df["tokens"] == 1000) & (df["dimension"] == 100000)])


def main():
    experiments = relic_helpers.load_experiments(filters, show_cmd=True)
    df = make_dataframe(experiments)

    print_table(df)


if __name__ == "__main__":
    main()
