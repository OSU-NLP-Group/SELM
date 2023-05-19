"""
Generates data for plotly.js histograms showing the distribution of an invidual feature for one or more pair of plaintexts.
"""

import argparse
import json

import numpy as np
import pandas as pd

from .. import attacking
from ..paper import helpers, security

files = ["News ($m1$)", "Rand. Bytes ($m2$)"]


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

    return parser


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


def main():
    parser = init_parser()
    args = parser.parse_args()

    ciphertexts = security.load_ciphertexts(args.group, 400)
    df = make_dataframe(ciphertexts)

    data = []
    for file in files:
        data.append(df[df.file == file]["l2-norm"].tolist())

    with open(f"docs/blog/data/{args.group}-histograms.json", "w") as fd:
        json.dump(data, fd)


if __name__ == "__main__":
    main()
