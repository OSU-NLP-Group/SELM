import pathlib

import datasets

from .. import util


def preprocess(output_dir):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    dataset = datasets.load_dataset(
        "reddit", cache_dir=util.HUGGINGFACE_CACHE_DIR
    ).shuffle(seed=42)["train"]["content"]

    indices = list(range(20))

    # 3 had a &nbsp;
    indices = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]

    posts = []
    for i in indices:
        posts.append(dataset[i])

    for i, post in enumerate(posts):
        with open(output_dir / f"{i}.txt", "w") as file:
            file.write(post.strip() + "\n")
