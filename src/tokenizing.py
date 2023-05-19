import argparse
import collections
import dataclasses
import math
import random
import uuid
from typing import Any, Dict, Iterator, List, Literal, Union

import datasets
import preface
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from . import config, logging, modeling, util

logger = logging.init(__name__)


DEFAULT_PROMPT = uuid.UUID("134b1f74-e9b9-4e94-96c5-2dfe56bc8a31")


def _join_encodings(*encs: Dict[str, object]) -> Dict[str, Any]:
    # check that all encodings have the same set of keys
    assert all(enc1.keys() == enc2.keys() for enc1, enc2 in zip(encs, encs[1:]))

    data: Dict[str, Any] = {}
    for key in encs[0]:
        if isinstance(encs[0][key], list):
            data[key] = sum((e[key] for e in encs), start=[])  # type: ignore
        elif isinstance(encs[0][key], torch.Tensor):
            data[key] = torch.concat(tuple(e[key] for e in encs), dim=1)  # type: ignore
        else:
            raise TypeError(f"Can't join encodings of type {type(encs[0][key])}")

    return data


@dataclasses.dataclass(frozen=True)
class Chunk:
    _prompt: str
    _text: str
    _eos_token: str

    def encoded(self, tokenizer, **tokenizer_kwargs) -> Dict[str, Any]:
        return _join_encodings(
            self.encoded_prompt(tokenizer, **tokenizer_kwargs),
            tokenizer(self._text + self._eos_token, **tokenizer_kwargs),
        )

    def encoded_prompt(self, tokenizer, **tokenizer_kwargs):
        return tokenizer(self._prompt, **tokenizer_kwargs)

    @property
    def text(self) -> str:
        return self._text


def load_chunks(text: str, data_config: config.DataConfig, tokenizer) -> List[Chunk]:
    """
    Given a raw text, create a list of Chunk objects that include prompt, chunk and eos token.

    This method sometimes modifies the tokenizer object by adding tokens to it.
    """

    if data_config.prompt_type == "token":
        prompts = _make_token_gen(tokenizer)
    elif data_config.prompt_type == "vocab":
        prompts = _make_vocab_gen(text, tokenizer)
    elif data_config.prompt_type == "uuid":
        prompts = _make_uuid_gen(1)
    elif data_config.prompt_type == "2x-uuid":
        prompts = _make_uuid_gen(2)
    elif data_config.prompt_type == "3x-uuid":
        prompts = _make_uuid_gen(3)
    elif data_config.prompt_type == "chunk-n":
        prompts = _make_chunk_n_gen()
    elif data_config.prompt_type == "natural-n":
        prompts = _make_natural_n_gen()
    elif data_config.prompt_type == "n-tokens":
        prompts = _make_n_tokens_gen(data_config.prompt_length, tokenizer)
    else:
        preface.never(data_config.prompt_type)

    return _make_text(prompts, text, data_config.chunk_length, tokenizer)


def _make_uuid_gen(n: int) -> Iterator[str]:
    """
    Returns n UUIDs at a time, joined by '-'

    Examples:

    next(_make_uuid_gen(1)) ->
    134b1f74-e9b9-4e94-96c5-2dfe56bc8a31

    next(_make_uuid_gen(2)) ->
    134b1f74-e9b9-4e94-96c5-2dfe56bc8a31-c398ef2b-55f6-3ebd-a2c1-1c1534c68068

    next(_make_uuid_gen(3)) ->
    134b1f74-e9b9-4e94-96c5-2dfe56bc8a31-c398ef2b-55f6-3ebd-a2c1-1c1534c68068-855ca571-2be8-3878-bb07-fe2316fb881d
    """
    last_uuid = DEFAULT_PROMPT
    uuids = [last_uuid]
    msg = "i-like-nlp"
    sep = "-"

    while True:
        while len(uuids) < n:
            last_uuid = uuid.uuid3(last_uuid, msg)
            uuids.append(last_uuid)

        yield sep.join(str(u) for u in uuids)
        uuids = []


def _make_token_gen(tokenizer):
    i = 0

    while True:
        token = f"<|start{i}|>"
        # add special tokens to tokenizer
        tokenizer.add_special_tokens({"additional_special_tokens": [token]})
        yield token
        i += 1


def _make_vocab_gen(text, tokenizer):
    used_tokens = set(tokenizer(text)["input_ids"])
    all_tokens = set(range(len(tokenizer)))
    unused_tokens = all_tokens - used_tokens - {tokenizer.eos_token_id}

    for unused in unused_tokens:
        yield tokenizer.decode(unused)


def _make_chunk_n_gen():
    i = 1

    while True:
        yield f"Chunk {i}: "
        i += 1


def _make_natural_n_gen():
    i = 0

    natural_ns = [
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "eleventh",
        "twelveth",
        "thirteenth",
        "fourteenth",
        "fifteenth",
        "sixteenth",
        "seventeenth",
        "eighteenth",
        "nineteenth",
        "twentieth",
    ]

    while True:
        yield f"The {natural_ns[i]} chunk is "
        i += 1


def _make_n_tokens_gen(n: int, tokenizer) -> Iterator[str]:
    assert n > 0, "Need at least one token!"
    all_tokens = list(range(len(tokenizer)))

    random.seed(n)

    while True:
        yield tokenizer.decode(random.choices(all_tokens, k=n))


def _make_text(
    prompts: Iterator[str],
    text: str,
    chunk_length: Union[Literal["longest"], int],
    tokenizer,
) -> List[Chunk]:
    """
    Returns a list of Chunk representing the original text, using the specified tokenizer.
    """

    if text == "":
        return []

    prompt = next(prompts)

    encoded_prompt = tokenizer(prompt)["input_ids"]
    encoded_text = tokenizer(text)["input_ids"]

    end = tokenizer.model_max_length - 1 - len(encoded_prompt)
    if isinstance(chunk_length, int):
        end = min(chunk_length - len(encoded_prompt), end)

    chunk = tokenizer.decode(encoded_text[:end], clean_up_tokenization_spaces=False)
    token_count = min(len(encoded_text[:end]), end)

    errors = 0
    while chunk != text[: len(chunk)]:
        # Some chinese characters require more than one token. So we either remove a character from the end and try again.
        errors += 1
        logger.info(
            "Removing a character from the end of chunk (probably a multi-token character)"
        )
        end -= 1
        chunk = tokenizer.decode(encoded_text[:end], clean_up_tokenization_spaces=False)
        token_count = min(len(encoded_text[:end]), end)

        if errors > 10:
            break

    assert chunk == text[: len(chunk)]

    text = text[len(chunk) :]

    logger.debug("Loaded chunk. [tokens: %s, chunk: %s]", token_count, chunk)

    return [Chunk(prompt, chunk, tokenizer.eos_token)] + _make_text(
        prompts, text, chunk_length, tokenizer
    )


def make_dataset(data_config: config.DataConfig, tokenizer):
    text = data_config.get_text()

    text_chunks = load_chunks(text, data_config, tokenizer)

    if "".join(chunk.text for chunk in text_chunks) != text:
        raise ValueError("Joining text chunks does not reassemble original text!")

    data = collections.defaultdict(list)

    for chunk in text_chunks:
        for key, value in chunk.encoded(tokenizer).items():
            data[key].append(value)

    tokenized_dataset = datasets.Dataset.from_dict(data)

    # don't bother padding for a single example.
    if len(tokenized_dataset) > 1:
        max_length = max(
            len(example[key]) for example in tokenized_dataset for key in example
        )

        def pad(example, max_length):
            assert "input_ids" in example
            assert "attention_mask" in example

            while len(example["input_ids"]) < max_length:
                example["input_ids"].append(tokenizer.eos_token_id)
            while len(example["attention_mask"]) < max_length:
                example["attention_mask"].append(0)

            return example

        tokenized_dataset = tokenized_dataset.map(
            lambda example: pad(example, max_length),
            batched=False,
            with_indices=False,
            load_from_cache_file=False,
            desc=f"Padding to length {max_length}",
        )

    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    return tokenized_dataset.map(
        add_labels,
        batched=True,
        load_from_cache_file=not data_config.overwrite_cache,
        desc="Adding labels",
    )


def measure_perplexity(
    model: torch.nn.Module, stride: int, input_ids: List[int], device
) -> float:
    max_length: int = model.config.max_position_embeddings  # type: ignore

    input_tensor = torch.tensor(input_ids).to(device)

    # negative log likelihoods
    nlls = []
    for i in tqdm(range(0, len(input_tensor), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, len(input_tensor))
        trg_len = end_loc - i  # may be different from stride on last loop
        inputs = input_tensor[begin_loc:end_loc].to(device)
        targets = inputs.clone().to(device)
        targets[:-trg_len] = -100

        with torch.no_grad():
            outputs = model(inputs, labels=targets)
            neg_log_likelihood = outputs[0] * trg_len
            nlls.append(neg_log_likelihood.item())

    logger.info("[Nans: %d]", len([i for i in nlls if math.isnan(i)]))
    nlls = [i for i in nlls if not math.isnan(i)]
    return torch.exp(torch.tensor(nlls).sum() / end_loc).item()


def new(exp_cfg: config.ExperimentConfig):
    if exp_cfg.tokenizer == "pretrained":
        tokenizer = AutoTokenizer.from_pretrained(
            exp_cfg.model.language_model_name_or_path,
            use_fast=True,
            cache_dir=util.HUGGINGFACE_CACHE_DIR,
        )
    elif exp_cfg.tokenizer == "1-byte":
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file="src/tokenizers/1-byte.json", eos_token="<|endoftext|>"
        )
    elif exp_cfg.tokenizer == "2-byte":
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file="src/tokenizers/2-byte.json", eos_token="<|endoftext|>"
        )
    else:
        preface.never(exp_cfg.tokenizer)

    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Experiment config to use.")
    parser.add_argument("--text", help="Text to tokenize")
    parser.add_argument("--files", help="File(s) to tokenize.", nargs="+", default=[])
    parser.add_argument(
        "--measure-perplexity",
        help="Whether to measure perplexity; requires a GPU.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--stride", help="Stride to use to measure perplexity", default=1, type=int
    )

    args = parser.parse_args()

    exp_config = next(config.load_configs(args.config))

    tokenizer = new(exp_config.tokenizer)

    texts = []

    if args.files:
        for file in args.files:
            with open(file, "r") as fd:
                texts.append((file, fd.read()))
    elif args.text:
        texts.append(("CLI", args.text))
    else:
        print("You must supply either --text or --files!")
        return

    if args.measure_perplexity:
        logger.info(
            "Loading model. [model: %s]",
            exp_config.model.language_model_name_or_path,
        )
        device = torch.device("cuda:0")
        model = modeling.new(exp_config.model, len(tokenizer), seed=42).to(device)

    for src, text in texts:
        chunks = load_chunks(text, exp_config.data, tokenizer)

        tokens = tokenizer(text)["input_ids"]

        print("source:", src)
        print("chunks:", len(chunks))
        print("tokens:", len(tokens))
        print("raw tokens:", tokens)
        if args.measure_perplexity:
            print("perplexity:", measure_perplexity(model, args.stride, tokens, device))


if __name__ == "__main__":
    cli()
