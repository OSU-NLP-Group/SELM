import itertools
import pathlib
from typing import Iterator

import datasets

from .. import util
from . import shared


def load_news_articles() -> Iterator[str]:
    dataset = datasets.load_dataset(
        "xsum", cache_dir=util.HUGGINGFACE_CACHE_DIR, streaming=True, split="validation"
    ).shuffle(seed=42)
    for example in dataset:
        document = example["document"]
        if not document:
            continue
        yield document


def preprocess(output_dir):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    articles = load_news_articles()

    example_count = 10
    token_lengths = (100, 300, 1000, 3000)

    for length, loc in itertools.product(token_lengths, range(example_count)):
        tokens = []
        text = ""
        while len(tokens) < length:
            text += next(articles)
            tokens = shared.tokenizer(text)["input_ids"]

        tokens = tokens[:length]
        shared.assert_invertible(tokens)

        text = shared.tokenizer.decode(tokens)

        length_dir = output_dir / f"{length}-tokens"
        length_dir.mkdir(exist_ok=True)

        with open(length_dir / f"{loc}.txt", "w") as file:
            file.write(text)

    char_lengths = (500, 2500, 5000, 25000)

    for length, loc in itertools.product(char_lengths, range(example_count)):
        text = ""
        while len(text) < length:
            text += next(articles)
        text = text[:length]

        tokens = shared.tokenizer(text)["input_ids"]
        shared.assert_invertible(tokens)
        text = shared.tokenizer.decode(tokens)

        length_dir = output_dir / f"{length}-chars"
        length_dir.mkdir(exist_ok=True)

        with open(length_dir / f"{loc}.txt", "w") as file:
            file.write(text)
