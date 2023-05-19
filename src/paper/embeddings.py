"""
Visualizes distributions of encrypted plaintexts using low-dimensional embedding techniques.
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.manifold
import sklearn.preprocessing
from tqdm.auto import tqdm

from . import helpers, security

sns.set_style("whitegrid", {"axes.grid": False})
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
        ],
        help="Which ciphertext groups to use.",
    )
    parser.add_argument(
        "count", type=int, help="How many experiments from each group to use"
    )
    parser.add_argument("filename", type=str, help="Filepath to save file to.")

    return parser


def plot_ciphertexts(ciphertexts):
    # Get point/file arrays
    # keys = sorted(ciphertexts.keys())
    keys = ["data/news/100-tokens/0.txt", "data/random-bytes/100-tokens/0.txt"]

    points = np.concatenate([ciphertexts[k] for k in keys])
    files = np.concatenate(
        [[helpers.translate_filename(k)] * len(ciphertexts[k]) for k in keys]
    )

    # Do dimension-reduction
    perplexity = 50
    learning_rate = 50
    trials = 5

    best_embedded = None
    best_divergence = np.inf

    # Try 10 random seeds
    for i in tqdm(range(trials)):
        # Ignore FutureWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            scaler = sklearn.preprocessing.StandardScaler()
            tsne = sklearn.manifold.TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=i,
                init="pca",
                learning_rate=learning_rate,
                n_iter=5000,
            )
            embedded = tsne.fit_transform(scaler.fit_transform(points))
            if tsne.kl_divergence_ < best_divergence:
                best_divergence = tsne.kl_divergence_
                best_embedded = embedded

    # Convert to dataframe
    rows = [
        (best_embedded[i][0], best_embedded[i][1], file) for i, file in enumerate(files)
    ]
    df = pd.DataFrame(rows, columns=["x", "y", "File"])

    order = ["News (N0)", "Rand. Bytes (RB)"]

    fig = sns.relplot(
        df,
        x="x",
        y="y",
        style="File",
        style_order=order,
        hue="File",
        hue_order=order,
        kind="scatter",
        facet_kws=dict(legend_out=False),
    )
    fig.set(xlabel=None, ylabel=None, xticks=[], yticks=[])
    fig.despine(right=True, top=True, bottom=True, left=True)
    fig.legend.set_title(None)
    sns.move_legend(fig, "upper right")

    return fig


def main():
    parser = init_parser()
    args = parser.parse_args()

    ciphertexts = security.load_ciphertexts(args.group, args.count)

    fig = plot_ciphertexts(ciphertexts)

    fig.savefig(args.filename, bbox_inches="tight")


if __name__ == "__main__":
    main()
