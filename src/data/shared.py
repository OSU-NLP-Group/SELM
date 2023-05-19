from typing import List, Sequence

from .. import config, tokenizing

exp_cfg = config.ExperimentConfig(
    model=config.ModelConfig(language_model_name_or_path="gpt2"),
    tokenizer="pretrained",
    data=config.DataConfig(__file__),
    training=config.TrainingConfig(),
)

tokenizer = tokenizing.new(exp_cfg)


def chunk_length(sequence: str) -> int:
    """
    Gets the length of a sequence in chunks using a GPT2 tokenizer.
    """
    if not sequence:
        return 0

    chunks = tokenizing.load_chunks(sequence, exp_cfg.data, tokenizer)

    return len(chunks)


def tokenize(sequence: str) -> List[int]:
    if not sequence:
        return []

    return tokenizer(sequence)["input_ids"]


def untokenize(tokens: Sequence[int]) -> str:
    return tokenizer.decode(tokens)


def token_length(sequence: str) -> int:
    """
    Gets the length of a sequence in tokens using a GPT2 tokenizer.
    """

    return len(tokenize(sequence))


def assert_invertible(tokens: List[int]):
    roundtrip_tokens = tokenize(untokenize(tokens))
    if tokens == roundtrip_tokens:
        return

    if untokenize(roundtrip_tokens) == untokenize(tokens):
        return

    print(untokenize(tokens))
    print(untokenize(roundtrip_tokens))

    for i, (t, rt) in enumerate(zip(tokens, roundtrip_tokens)):
        if t == rt:
            continue

        breakpoint()
