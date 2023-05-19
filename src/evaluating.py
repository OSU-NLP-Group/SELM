"""
Given a model and some text, check if the model is finished (correctly decodes the text from the prompt)
"""
import dataclasses
import logging
from typing import List

import numpy as np

from . import accelerate, config, tokenizing

TOKENIZER_KWARGS = {"add_special_tokens": False, "return_tensors": "pt"}

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Metrics:
    passed: bool
    first_failed_token: int
    reason: str
    gen_toks: List[int]
    tgt_toks: List[int]
    gen_text: str
    tgt_text: str


def passes(
    model, tokenizer, experiment_config: config.ExperimentConfig, epochs: int
) -> Metrics:
    # If there is a warmup value, and we have not warmed up all the way yet,
    # we can't be finished.
    if experiment_config.training.regularization.warmup:
        if epochs + 1 < experiment_config.training.regularization.warmup:
            return Metrics(False, -1, "not finished warming up", [], [], "", "")

    raw_text = experiment_config.data.get_text()

    chunks = tokenizing.load_chunks(raw_text, experiment_config.data, tokenizer)

    model.eval()

    for i, chunk in enumerate(chunks):
        prompt = chunk.encoded_prompt(tokenizer, **TOKENIZER_KWARGS)["input_ids"]

        tgt_toks = chunk.encoded(tokenizer, **TOKENIZER_KWARGS)["input_ids"]
        tgt_toks = tgt_toks.squeeze_().tolist()
        tgt_text = chunk.text

        model_max_len = 0
        if hasattr(model.config, "max_position_embeddings"):
            model_max_len = model.config.max_position_embeddings
        elif hasattr(model.config, "n_positions"):
            model_max_len = model.config.n_positions
        assert model_max_len > 0

        max_length = min(model_max_len, len(tgt_toks) + np.prod(prompt.shape) + 2)

        input_ids = prompt.to(accelerate.input_device)

        logger.info(f"Generating output sequence. [chunk: {i}]")
        gen_toks = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=experiment_config.model.temperature,
            top_k=experiment_config.model.top_k,
            top_p=experiment_config.model.top_p,
            repetition_penalty=experiment_config.model.repetition_penalty,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=model.config.eos_token_id,
        )

        logger.info("Doing post-processing on output sequence.")
        gen_toks = gen_toks.squeeze_().tolist()

        # Conditionally log the output sequence if it is not too long.
        gen_token_str = "too long"
        if len(gen_toks) < 800:
            gen_token_str = repr(gen_toks)
        logger.info(
            "Generated output sequence. [length: %d, gen_toks: %s]",
            len(gen_toks),
            gen_token_str,
        )

        # Remove all text after the eos token (including the eos tokens)
        if tokenizer.eos_token_id in gen_toks:
            gen_toks = gen_toks[: gen_toks.index(tokenizer.eos_token_id)]

        # Decode text
        gen_text = tokenizer.decode(gen_toks, clean_up_tokenization_spaces=False)

        # Remove the prompt at the beginning of the sequence.
        gen_text = gen_text[len(tokenizer.decode(prompt[0])) :]

        success = gen_text == tgt_text

        if success:
            logger.info(f"Successfully generated chunk! [chunk: {i}]")

        if not success:
            # Find first failed token token
            for first_fail, (gen, tgt) in enumerate(zip(gen_toks, tgt_toks)):
                if gen != tgt:
                    break

            logger.info(
                f"For chunk {i}, generated text: '{gen_text}'\n\ndoes not match target text: '{tgt_text}'.\n\nGenerated tokens: {gen_toks}\nTarget tokens:    {tgt_toks}\nFirst failed token: {first_fail}."
            )
            return Metrics(
                False, first_fail, f"chunk {i}", gen_toks, tgt_toks, gen_text, tgt_text
            )

    logger.info(f"Successfully decoded text from {len(chunks)} chunk(s)!")
    return Metrics(True, -1, "success!", [], [], "", "")
