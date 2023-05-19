import tempfile

from .. import config, tokenizing


class DummyTokenizer:
    eos_token = ord("~")

    def __init__(self, model_max_length):
        self.model_max_length = model_max_length

    def __call__(self, text):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


def test_load_chunks_smoke():
    text = "hello world!"
    with tempfile.NamedTemporaryFile() as data_file:
        data_file.write(text.encode())
        data_config = config.DataConfig(data_file.name)

    tokenizer = DummyTokenizer(100)
    actual = tokenizing.load_chunks(text, data_config, tokenizer)
    expected = [
        tokenizing.Chunk(str(tokenizing.DEFAULT_PROMPT), text, tokenizer.eos_token)
    ]

    assert actual == expected
