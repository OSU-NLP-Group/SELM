"""
Demonstrates that ciphertexts have approximately normal distributions
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()

from . import security

sns.set_style("whitegrid", {"axes.grid": False})
sns.set_context("paper", font_scale=0.7)
sns.set_palette("Dark2")


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Filepath to save file to.")

    return parser


def main():
    parser = init_parser()
    args = parser.parse_args()

    ciphertexts = security.load_ciphertexts("original", count=2)

    bound = 5e-7

    fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True)
    bins = np.linspace(start=-bound, stop=bound, num=30)

    axes[0][0].hist(ciphertexts["data/news/100-tokens/0.txt"][0], bins=bins)
    axes[0][1].hist(ciphertexts["data/news/100-tokens/0.txt"][1], bins=bins)
    axes[0][0].set_ylabel("News (0)")

    axes[1][0].hist(ciphertexts["data/news/100-tokens/1.txt"][0], bins=bins)
    axes[1][1].hist(ciphertexts["data/news/100-tokens/1.txt"][1], bins=bins)
    axes[1][0].set_ylabel("News (1)")

    axes[2][0].hist(ciphertexts["data/pubmed/100-tokens/0.txt"][0], bins=bins)
    axes[2][1].hist(ciphertexts["data/pubmed/100-tokens/0.txt"][1], bins=bins)
    axes[2][0].set_ylabel("PubMed")

    axes[3][0].hist(ciphertexts["data/random-words/100-tokens/0.txt"][0], bins=bins)
    axes[3][1].hist(ciphertexts["data/random-words/100-tokens/0.txt"][1], bins=bins)
    axes[3][0].set_ylabel("Rand. Words")

    axes[4][0].hist(ciphertexts["data/random-bytes/100-tokens/0.txt"][0], bins=bins)
    axes[4][1].hist(ciphertexts["data/random-bytes/100-tokens/0.txt"][1], bins=bins)
    axes[4][0].set_ylabel("Rand. Bytes")

    axes[0][0].set(yticklabels=[])

    fig.tight_layout()
    fig.savefig(args.filename, bbox_inches="tight")


if __name__ == "__main__":
    main()
