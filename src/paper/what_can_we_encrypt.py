"""
Makes the charts for what-can-we-encrypt experiments.

There are three figures.
For all three figures, the x-axis is the intrinsic dimension and the y-axis is the number of epochs.
There are multiple lines on each figure, which show the effect of a third variable.

1. Measuring the effects of different domains (100 tokens, gpt2)
2. Measuring the effects of different lengths (news domain, gpt2)
3. Measuring the effects of different models (1000 tokens, news domain)
"""
import argparse
from typing import List

import pandas as pd
import relic
import seaborn as sns

from .. import relic_helpers
from . import helpers

sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.2)
sns.set_palette("Dark2")


domain_filters = (
    "(== training.regularization.variety None)",
    "(== training.lr_scheduler_type 'linear-const')",
    "(== training.plateau_reduce_factor 0.1)",
    "(== tokenizer.pretrained 'gpt2')",
    "(== data.prompt_length None)",
    # Fix model size and length
    "(== model.language_model_name_or_path 'gpt2')",
    "(~ data.file '100-tokens')",
    "(or (~ data.file 'news') (~ data.file 'pubmed') (~ data.file 'random') (~ data.file 'dog-pic-'))",
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

length_filters = (
    "(== training.regularization.variety None)",
    "(== training.lr_scheduler_type 'linear-const')",
    "(== training.batch_size 2)",
    "(== training.plateau_reduce_factor 0.1)",
    "(== data.prompt_length None)",
    "(== tokenizer.pretrained 'gpt2')",
    # Fix these two variables (model size and domain)
    "(== model.language_model_name_or_path 'gpt2')",
    "(~ data.file 'news')",
    # Bad experiments
    "(not (~ experimenthash 'f318516b11'))",  # Succeeded but was an outlier (other two failed)
    # Handle different learning rate for 100K
    r"""(or
            (and
                (< model.intrinsic_dimension 100000)
                (== training.learning_rate 0.00000002))
            (and
                (== model.intrinsic_dimension 100000)
                (== training.learning_rate 0.00000001)))""",
)

model_filters = (
    "(~ data.file 'news')",
    "(~ data.file '100-tokens')",
    "(== training.regularization.variety None)",
    "(== training.lr_scheduler_type 'linear-const')",
    "(== training.plateau_reduce_factor 0.1)",
    "(== training.batch_size 1)",
    "(== data.prompt_length 0)",
    "(== tokenizer 'pretrained')",
    r"""
    (or
        (and
            (== model.language_model_name_or_path 'gpt2')
            (== training.clipping.value 100000)
            (< model.intrinsic_dimension 100000)
            (== training.learning_rate 0.00000002))
        (and
            (== model.language_model_name_or_path 'gpt2')
            (== training.clipping.value 100000)
            (== model.intrinsic_dimension 100000)
            (== training.learning_rate 0.00000001))
        (== model.language_model_name_or_path 'cerebras/Cerebras-GPT-111M'))"""
    # (and
    # (== model.language_model_name_or_path 'gpt2-medium')
    # (== training.clipping.value 10000)
    # (< model.intrinsic_dimension 100000)
    # (== training.learning_rate 0.00000002))
    # (and
    # (== model.language_model_name_or_path 'gpt2-medium')
    # (== training.clipping.value 10000)
    # (== model.intrinsic_dimension 100000)
    # (== training.learning_rate 0.00000001)))"""
)


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("fig", choices=["model", "domain", "length"])
    parser.add_argument("filename", type=str, help="Filepath to save file to.")
    parser.add_argument("--dpi", type=int, help="DPI for saved PNGs", default=400)

    return parser


def make_dataframe(experiments: List[relic.Experiment]) -> pd.DataFrame:
    headers = ["Length", "Domain", "Dimension", "Model", "Epochs"]

    groups = set()

    rows = []
    for exp in experiments:
        length = helpers.parse_length(exp.config["data"]["file"])
        dimension = exp.config["model"]["intrinsic_dimension"]
        domain = helpers.translate_domain(
            helpers.parse_domain(exp.config["data"]["file"])
        )
        model = helpers.translate_model(
            exp.config["model"]["language_model_name_or_path"],
            exp.config["model"]["pretrained"],
        )

        if not len(exp):
            print("skipping experiment with no trials", exp.hash)
            continue

        if len(exp) > 1:
            print(exp.hash, len(exp))

        trial = exp[0]
        if not trial["finished"]:
            continue

        epochs = trial["epochs"]

        rows.append([length, domain, dimension, model, epochs])
        groups.add((length, domain, dimension, model))

    # There are duplicate rows in here if you ignore epochs
    # If for a given set of (length, domain, dimension, model), all the epoch
    # values are 10K, then it never succeeded and it should be removed.
    df = pd.DataFrame(data=rows, columns=headers)

    for length, domain, dimension, model in groups:
        rows = df[
            (df.Length == length)
            & (df.Domain == domain)
            & (df.Dimension == dimension)
            & (df.Model == model)
        ]

        if len(rows) > 10:
            breakpoint()
            raise ValueError("Internal logic error; adjust the filters?")

        if all(epoch == 10_000 for epoch in rows.Epochs):
            df = df.drop(rows.index)

    return df


def plot_model(df):
    return sns.relplot(
        df,
        x="Dimension",
        y="Epochs",
        hue="Model",
        kind="line",
        palette="Dark2",
        hue_order=[
            "GPT-2",
            "GPT-2 (rand)",
            "Cerebras",
        ],
        facet_kws=dict(legend_out=False),
        aspect=1.3,
    )


def plot_length(df):
    return sns.relplot(
        df,
        x="Dimension",
        y="Epochs",
        hue="Length",
        kind="line",
        palette="Dark2",
        hue_order=[100, 300, 1000, 3000],
        facet_kws=dict(legend_out=False),
        aspect=1.3,
    )


def plot_domain(df):
    return sns.relplot(
        df,
        x="Dimension",
        y="Epochs",
        hue="Domain",
        kind="line",
        palette="Dark2",
        hue_order=["News", "PubMed", "Random Words", "Random Bytes"],
        facet_kws=dict(legend_out=False),
        aspect=1.3,
    )


def adjust_fig(fig):
    fig.set(ylim=(10**0.87, 10**4.13))
    fig.set(xscale="log", yscale="log")
    sns.move_legend(fig, "upper right")


def main():
    parser = init_parser()
    args = parser.parse_args()

    if args.fig == "model":
        experiments = relic_helpers.load_experiments(model_filters, show_cmd=True)
    elif args.fig == "length":
        experiments = relic_helpers.load_experiments(length_filters, show_cmd=True)
    elif args.fig == "domain":
        experiments = relic_helpers.load_experiments(domain_filters, show_cmd=True)

    df = make_dataframe(experiments)

    if args.fig == "model":
        fig = plot_model(df)
    elif args.fig == "length":
        fig = plot_length(df)
    elif args.fig == "domain":
        fig = plot_domain(df)

    adjust_fig(fig)

    fig.tight_layout()
    fig.savefig(args.filename, bbox_inches="tight", dpi=args.dpi)


if __name__ == "__main__":
    main()
