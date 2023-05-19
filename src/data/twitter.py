import pathlib
import random

import datasets

from .. import util


def preprocess(output_dir):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # load all tweets
    all_tweets = []
    for subset in ["emotion", "sentiment"]:
        dataset = datasets.load_dataset(
            "tweet_eval", subset, cache_dir=util.HUGGINGFACE_CACHE_DIR
        )
        for tweet in dataset["train"]["text"]:
            all_tweets.append(tweet)

    # pick ten tweets.
    tweets = random.choices(all_tweets, k=10)

    # write them to disk
    for i, tweet in enumerate(tweets):
        with open(output_dir / f"{i}.txt", "w") as file:
            file.write(tweet + "\n")
