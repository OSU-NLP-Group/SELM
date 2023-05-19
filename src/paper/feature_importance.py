"""
This script measure the feature importance for each proposed variant of the encryption algorithm.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.feature_selection

from .. import attacking, logging
from . import helpers, security

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=2)
sns.set_palette("Dark2")

logger = logging.init("feature-importance")


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--relics",
        help="Path to relics/ directory",
    )
    parser.add_argument(
        "count", type=int, help="How many experiments from each group to use"
    )
    parser.add_argument("filename", type=str, help="Filepath to save file to.")

    return parser


def measure_mutual_information(datasets):
    files = ("data/news/100-tokens/0.txt", "data/random-bytes/100-tokens/0.txt")
    datasets = [dataset for dataset in datasets if dataset.name in files]
    # Arrange the datasets into a single X, y multiclass classification problem.
    x = np.concatenate([dataset.splits[0] for dataset in datasets], axis=0)
    y = np.zeros(x.shape[0])
    start = 0
    end = 0
    for i, dataset in enumerate(datasets):
        start = end
        end += dataset.splits[0].shape[0]
        y[start:end] = i

    # Measure mutual information
    mi = sklearn.feature_selection.mutual_info_classif(
        x,
        y,
        discrete_features=False,
        n_neighbors=3,
        copy=True,
        random_state=42,
    )

    return mi


def load_datasets(group, *, count):
    ciphertexts = security.load_ciphertexts(group, count=count)
    return list(
        attacking.data.make_single_datasets(
            ciphertexts, attacking.data.preprocess, ratio=1.0
        )
    )


def plot_mi(original, l2_norm, dist, keys):
    fig, ax = plt.subplots(subplot_kw={"aspect": 9})

    x = np.arange(len(keys))
    width = 0.3

    ax.bar(x - width, original, width, label="Original")
    ax.bar(x, l2_norm, width, label="L2-Norm Reg.")
    ax.bar(x + width, dist, width, label="Dist. Reg.")
    ax.set_xticks(x)
    ax.set_xticklabels([helpers.translate_feature(k) for k in keys])
    ax.set_ylabel("Mutual Information")
    ax.set_xlabel("Feature")
    ax.legend()

    return fig


def main():
    parser = init_parser()
    args = parser.parse_args()

    # Load ciphertexts from experiments
    original_mi = measure_mutual_information(
        load_datasets("original", count=args.count)
    )
    l2_norm_mi = measure_mutual_information(
        load_datasets("l2-norm-reg", count=args.count)
    )
    dist_mi = measure_mutual_information(
        load_datasets("distribution-reg", count=args.count)
    )
    keys = sorted(attacking.data.FEATURE_FUNCTIONS.keys())

    fig = plot_mi(original_mi, l2_norm_mi, dist_mi, keys)
    fig.tight_layout()
    fig.savefig(args.filename, bbox_inches="tight")


if __name__ == "__main__":
    main()
