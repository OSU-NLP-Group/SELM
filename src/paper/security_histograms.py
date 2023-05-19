"""
Generates histograms showing the distribution of an invidual feature for one or more pair of plaintexts.
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .. import attacking
from . import helpers, security

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.3)
sns.set_palette("Dark2")


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "group",
        choices=[
            "original",
            "l2-norm-reg",
            "distribution-reg",
            "perplexity-bounded",
        ],
        help="Which ciphertext groups to use.",
    )
    parser.add_argument(
        "--x-func",
        choices=list(attacking.data.FEATURE_FUNCTIONS),
        help="Which feature function to use on the x-axis.",
        required=True,
    )
    parser.add_argument(
        "--y-func",
        choices=list(attacking.data.FEATURE_FUNCTIONS),
        help="Which feature function to use on the y-axis.",
    )
    parser.add_argument(
        "count", type=int, help="How many experiments from each group to use"
    )
    parser.add_argument("filename", type=str, help="Filepath to save file to.")

    return parser


def get_hue_order(files):
    return ["News ($m1$)", "Rand. Bytes ($m2$)"]


def make_dataframe(ciphertexts):
    headers = ["file", *attacking.data.FEATURE_FUNCTIONS.keys()]

    rows = []

    for file, matrix in ciphertexts.items():
        features = {}

        for name, func in attacking.data.FEATURE_FUNCTIONS.items():
            features[name] = func(matrix)

        # features[X] are all the same length
        features["file"] = [
            helpers.translate_filename(file) for _ in range(len(features[name]))
        ]

        file_rows = tuple(
            [features[key][i] for key in headers] for i in range(len(features[name]))
        )

        rows.extend(file_rows)

    return pd.DataFrame.from_records(data=rows, columns=headers)


def plot_2d_distribution(df, x, y):
    hue_order = get_hue_order(set(df["file"]))

    fig = sns.relplot(
        df,
        y=y,
        x=x,
        hue="file",
        hue_order=hue_order,
        # Makes it X times wide as it is tall
        aspect=2,
        # Makes the legend stay inside the figure area
        facet_kws=dict(legend_out=False),
    )

    sns.move_legend(fig, "upper left", frameon=True)
    fig.set(ylabel=helpers.translate_feature(y), xlabel=helpers.translate_feature(x))

    return fig


def plot_1d_distribution(df, x):
    hue_order = get_hue_order(set(df["file"]))

    fig = sns.displot(
        df,
        x=x,
        hue="file",
        hue_order=hue_order,
        kind="kde",
        bw_adjust=0.6,
        # Fill the area under each curve
        fill=True,
        # Makes it X times wide as it is tall
        aspect=2,
        # Makes the legend stay inside the figure area
        facet_kws=dict(legend_out=False),
    )
    sns.move_legend(fig, "upper right", frameon=True)
    fig.set(xlabel=helpers.translate_feature(x), yticklabels=[])

    return fig


def main():
    parser = init_parser()
    args = parser.parse_args()

    ciphertexts = security.load_ciphertexts(args.group, args.count)
    df = make_dataframe(ciphertexts)

    if args.y_func is None:
        # Doing a 1D histogram
        fig = plot_1d_distribution(df, args.x_func)
    else:
        # Have a args.y_func, doing 2D distribution
        fig = plot_2d_distribution(df, args.x_func, args.y_func)

    fig.savefig(args.filename)


if __name__ == "__main__":
    main()
