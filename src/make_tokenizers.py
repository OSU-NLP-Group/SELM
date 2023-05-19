import tokenizers


def main():
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())

    # 1-byte.json
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
        add_prefix_space=False, trim_offsets=True, use_regex=False
    )

    trainer = tokenizers.trainers.BpeTrainer(
        special_tokens=["<|endoftext|>"],
        initial_alphabet=alphabet,
        vocab_size=len(alphabet),
    )
    tokenizer.train([], trainer)
    tokenizer.save("src/tokenizers/1-byte.json")

    # 2-byte tokenizer
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
        add_prefix_space=False, trim_offsets=True, use_regex=False
    )

    trainer = tokenizers.trainers.BpeTrainer(
        special_tokens=["<|endoftext|>"],
        initial_alphabet=alphabet,
        vocab_size=len(alphabet) * (len(alphabet) + 1) + 1,
    )

    data = [i + j for i in alphabet for j in alphabet]

    tokenizer.train_from_iterator(data, trainer)
    tokenizer.save("src/tokenizers/2-byte.json")


if __name__ == "__main__":
    main()
