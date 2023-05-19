import argparse
import os

import torch

from src import accelerate, config, logging, modeling, tokenizing

logger = logging.init("SELM")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="+", help="File(s) to encrypt")
    parser.add_argument(
        "--key",
        required=True,
        type=int,
        help="Symmetric key to use as a seed. You can generate a key with:\n\n\tpython -c 'import secrets; print(secrets.randbits(32))'",
    )
    parser.add_argument(
        "--int-dim", type=int, help="Intrinsic dimension.", default=10_000
    )

    return parser.parse_args()


def new_cfg(int_dim, file):
    return config.ExperimentConfig(
        model=config.ModelConfig(
            "gpt2",
            int_dim,
            dropout=0.0,
            int_dim_dropout=0.0,
            normalized=False,
        ),
        tokenizer="pretrained",
        data=config.DataConfig(file),
        training=config.TrainingConfig(
            maximum_epochs=10_000,
            learning_rate=2e-8,
            report_interval=10,
            lr_scheduler_type="linear-const",
            warmup_epochs=0,
            decay_epochs=2000,
            decayed_proportion=0.1,
            clipping=config.ClippingConfig(
                algorithm="norm",
                value=1e5,
            ),
            regularization=config.RegularizationConfig(
                variety="distribution-difference-integral",
                weight=5e8,
                mean=0,
                std=4e-7,
                schedule="linear",
                warmup=500,
            ),
        ),
        trials=1,
    )


def get_encoded_prompt(data_cfg, tokenizer):
    chunk = tokenizing.load_chunks("dummy-text", data_cfg, tokenizer)[0]
    prompt = chunk.encoded_prompt(
        tokenizer, add_special_tokens=False, return_tensors="pt"
    )["input_ids"]
    return prompt


if __name__ == "__main__":
    args = parse_args()

    for file in args.file:
        vec = torch.load(file)
        int_dim = len(vec)

        exp_cfg = new_cfg(int_dim, file)
        tokenizer = tokenizing.new(exp_cfg)
        model = modeling.new(exp_cfg.model, vocab=len(tokenizer), seed=args.key)
        with torch.no_grad():
            model.set_intrinsic_dimension_vector(vec)

        model = accelerate.prepare(model)

        # prompt ids
        prompt = get_encoded_prompt(exp_cfg.data, tokenizer)
        input_ids = prompt.to(accelerate.input_device)

        max_length = 0
        if hasattr(model.config, "max_position_embeddings"):
            max_length = model.config.max_position_embeddings
        elif hasattr(model.config, "n_positions"):
            max_length = model.config.n_positions
        assert max_length > 0

        logger.info("Generating message.")
        gen_toks = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=exp_cfg.model.temperature,
            top_k=exp_cfg.model.top_k,
            top_p=exp_cfg.model.top_p,
            repetition_penalty=exp_cfg.model.repetition_penalty,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=model.config.eos_token_id,
        )

        logger.info("Doing post-processing on output sequence.")
        gen_toks = gen_toks.squeeze_().tolist()

        # Remove all text after the eos token (including the eos tokens)
        if tokenizer.eos_token_id in gen_toks:
            gen_toks = gen_toks[: gen_toks.index(tokenizer.eos_token_id)]

        # Decode text
        gen_text = tokenizer.decode(gen_toks, clean_up_tokenization_spaces=False)

        # Remove the prompt at the beginning of the sequence.
        gen_text = gen_text[len(tokenizer.decode(prompt[0])) :]

        out_filename = f"{os.path.basename(file)}.dec".removesuffix(".enc.dec")
        with open(out_filename, "w") as fd:
            fd.write(gen_text)
