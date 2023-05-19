import pathlib

import datasets

from .. import util


def preprocess(output_dir):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    openwebtext = datasets.load_dataset(
        "stas/openwebtext-10k", split="train", cache_dir=util.HUGGINGFACE_CACHE_DIR
    ).shuffle(seed=42)

    for i in range(10):
        article = openwebtext[i]
        with open(output_dir / f"{i}.txt", "w") as file:
            file.write(article["text"].strip() + "\n")
