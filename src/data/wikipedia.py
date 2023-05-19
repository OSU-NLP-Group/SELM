import itertools
import pathlib

import datasets
from tqdm.auto import tqdm

from .. import util
from . import shared


def preprocess(output_dir):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    wikipedia = datasets.load_dataset(
        "wikipedia", "20200501.en", split="train", cache_dir=util.HUGGINGFACE_CACHE_DIR
    ).shuffle(seed=42)

    lengths = (100, 200, 500, 996)

    articles = iter(wikipedia)

    for length, loc in tqdm(itertools.product(lengths, range(10))):
        tokens = []
        while len(tokens) < length:
            article = next(articles)
            tokens = shared.tokenizer(article["text"])["input_ids"]

        tokens = tokens[:length]
        shared.assert_invertible(tokens)

        text = shared.tokenizer.decode(tokens)
        assert shared.chunk_length(text) == 1

        length_dir = output_dir / f"{length}-tokens"
        length_dir.mkdir(exist_ok=True)

        with open(length_dir / f"{loc}.txt", "w") as file:
            file.write(text)
