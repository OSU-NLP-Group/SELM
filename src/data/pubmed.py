import itertools
import pathlib
from typing import Iterator

import datasets

from .. import util
from . import shared


def take(n, iterable):
    "Return first n items of the iterable"
    return itertools.islice(iterable, n)


def load_pubmed_abstracts() -> Iterator[str]:
    dataset = datasets.load_dataset(
        "pubmed", cache_dir=util.HUGGINGFACE_CACHE_DIR, streaming=True, split="train"
    ).shuffle(seed=42)
    for example in dataset:
        abstract = example["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
        if not abstract:
            continue
        yield abstract


def preprocess(output_dir):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    abstracts = load_pubmed_abstracts()

    example_count = 10
    lengths = (100, 1_000, 10_000)

    for length, loc in itertools.product(lengths, range(example_count)):
        tokens = []
        text = ""
        while len(tokens) < length:
            text += next(abstracts)
            tokens = shared.tokenizer(text)["input_ids"]

        tokens = tokens[:length]
        shared.assert_invertible(tokens)

        text = shared.tokenizer.decode(tokens)

        length_dir = output_dir / f"{length}-tokens"
        length_dir.mkdir(exist_ok=True)

        with open(length_dir / f"{loc}.txt", "w") as file:
            file.write(text)
