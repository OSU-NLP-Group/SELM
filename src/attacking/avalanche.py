"""
Measures the avalanche effect of our algorithm.
"""
import argparse
import os
import re
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import relic
import torch
from tqdm.auto import tqdm

CIPHERTEXT_DIMENSION = 10_000
MESSAGE_LENGTH = 100
PATTERN = re.compile(r"incrementing/100-token/uniform/\d\d?/")


def save_heatmap(matrix, path):
    vmin, vmax = 0, 4e-7
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    # Plot heatmap
    # Reshape for better visualization
    im = ax1.imshow(matrix.reshape(1_000, 1_000), norm=norm)
    fig.colorbar(im, ax=ax1)

    # Plot distribution
    bins = np.linspace(vmin, vmax, num=100)
    ax2.hist(matrix.reshape(10_000, -1).squeeze(), bins=bins)

    fig.savefig(path)


def load_ciphertext(exp):
    saved = torch.load(exp.model_path(0))
    return saved["theta_d"].numpy()


def make_filter_fn(fixed_key):
    def filter_fn(exp):
        file_match = PATTERN.search(exp.config["data"]["file"])

        if fixed_key:
            return exp.config["seed"] == 0 and file_match
        else:
            return file_match and (
                exp.config["seed"] > 0
                or exp.config["data"]["file"]
                == "data/incrementing/100-token/uniform/0/original.txt"
            )

    return filter_fn


def load_ciphertext_lookup(fixed_key):
    lookup = {}
    error_count = 0
    for exp in tqdm(relic.load_experiments(filter_fn=make_filter_fn(fixed_key))):
        try:
            assert exp.model_exists(0)
            lookup[exp.config["data"]["file"]] = load_ciphertext(exp)
        except (EOFError, AssertionError):
            print(exp.hash, exp.config)
            error_count += 1

    print("Errors:", error_count)

    print("Loaded ciphertexts:", len(lookup))

    return lookup


def get_vi(
    plaintext_file: str, i: int, lookup: Dict[str, np.array], allow_missing=False
):
    """
    Gets the difference in ciphertexts between the plaintext and the plaintext incremented at i.
    """
    try:
        # 1. Get the ciphertext associated with the plaintext
        x = lookup[plaintext_file]

        # 2. Get the ciphertext associated with the plaintext incremented at i
        root = os.path.dirname(plaintext_file)
        incremented_file = os.path.join(root, f"{i}.txt")
        x_i = lookup[incremented_file]
    except KeyError:
        if allow_missing:
            return np.zeros((10_000,))

        raise

    return np.abs(x - x_i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixed-key",
        action="store_true",
        help="Whether to use the fixed key experiments",
        default=False,
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Whether to complete the experiment even if there are missing ciphertexts",
        default=False,
    )
    args = parser.parse_args()

    lookup = load_ciphertext_lookup(args.fixed_key)

    A = np.zeros((CIPHERTEXT_DIMENSION, MESSAGE_LENGTH))

    for j in tqdm(range(100)):
        filepath = f"data/incrementing/100-token/uniform/{j}/original.txt"

        A += np.array(
            [
                get_vi(filepath, i, lookup, allow_missing=args.allow_missing)
                for i in range(MESSAGE_LENGTH)
            ]
        ).T

    A = A / len(lookup) * 100

    print(A.shape, np.mean(A), np.std(A), np.max(A), np.min(A))

    if args.fixed_key:
        np.save("data/cached/avalanche-matrix-fixed-key", A)
        save_heatmap(A, "fixed-key.pdf")
    else:
        np.save("data/cached/avalanche-matrix-random-key", A)
        save_heatmap(A, "random-key.pdf")


if __name__ == "__main__":
    main()
