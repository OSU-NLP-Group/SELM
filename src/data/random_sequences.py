"""
From all the downloaded data, creates random sequences of STUFF.
"""
import itertools
import logging
import os
import pathlib
import random
from typing import AbstractSet, Callable, List, Sequence, Set, Union

import datasets
from tqdm.auto import tqdm, trange

from .. import util
from . import shared

logger = logging.getLogger(__name__)


BAD_EXTENSIONS = {".pckl"}

EXAMPLE_COUNT = 10


def wikipedia_sampling(processing_fn: Callable[[str], Set[str]]) -> Set[str]:
    # If you get errors at this line, look at this issue: https://github.com/huggingface/datasets/pull/5321
    # The fix is just pip install datasets==2.6.1
    wikipedia = datasets.load_dataset(
        "wikipedia", "20220301.en", split="train", cache_dir=util.HUGGINGFACE_CACHE_DIR
    ).shuffle(seed=42)

    results: Set[str] = set()

    for i in trange(10**4):
        text: str = wikipedia[i]["text"]  # type: ignore
        results = results.union(processing_fn(text))

    return results


def load_from_files(
    input_dirs: List[pathlib.Path], processing_fn: Callable[[str], Set[str]]
) -> Set[str]:
    """
    Reads all files in input_dir (except for those with BAD_EXTENSIONS) and applies processing_fn to the text contents.

    Returns a union of the results from processing_fn.
    """

    results: Set[str] = set()

    for input_dir in input_dirs:
        for dirpath, _, filenames in tqdm(sorted(os.walk(input_dir))):
            for filename in filenames:
                root, ext = os.path.splitext(filename)

                if ext in BAD_EXTENSIONS:
                    continue

                filepath = os.path.join(dirpath, filename)

                with open(filepath) as file:
                    text = file.read()

                results = results.union(processing_fn(text))

    return results


def generate_random_sequence(
    sample_space: Union[Sequence[str], AbstractSet[str]],
    max_tokens: int,
    with_replacement: bool = False,
    separator: str = " ",
):
    """
    Generates a random sequence of items from sample_space. The sequence length respects the limits set by max_tokens or max_chunks, in that order.

    Args:
        sample_space: a bunch of strings to sample from.
        max_tokens: maximum number of tokens in the returned length.
        with_replacement: whether to sample from sample_space with replacement.
        separator: a string to separate samples from sample_space.

    [IMPLEMENTATION DETAILS]

    When generating a random sequence of letters or words, it can be very time-consuming to tokenize the input once for every letter in a sequence that is supposed to be 4 or 8 chunks long.

    To efficiently find the right sequence length:
    1. The sequence length is doubled until it is over the limit.
    2. Tokenize the sequence and take the [0:max_tokens]
    """
    shuffled: List[str] = random.sample(sample_space, len(sample_space))

    def get_next():
        nonlocal shuffled
        if with_replacement:
            return random.choice(shuffled)

        try:
            return shuffled.pop()
        except IndexError:
            shuffled = random.sample(sample_space, len(sample_space))
            return get_next()

    def current_sequence():
        return separator.join(sequence)

    sequence: List[str] = [get_next()]

    # Double sequence length until it's too long
    while shared.token_length(current_sequence()) <= max_tokens:
        old_length = len(sequence)
        for _ in range(old_length):
            sequence.append(get_next())

    # Cut off the excess tokens
    tokens = shared.tokenize(current_sequence())[:max_tokens]

    shared.assert_invertible(tokens)
    return shared.untokenize(tokens)


def make_data(output_dir, sample_space: Set[str], **generate_random_sequence_kwargs):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    lengths = (100, 1_000, 10_000)

    for length, i in tqdm(list(itertools.product(lengths, range(EXAMPLE_COUNT)))):
        folder = output_dir / f"{length}-tokens"
        folder.mkdir(exist_ok=True)

        example = generate_random_sequence(
            sample_space, max_tokens=length, **generate_random_sequence_kwargs
        )
        with open(folder / f"{i}.txt", "w") as file:
            file.write(example)
