import argparse
import logging
import string
import sys

log_format = "[%(levelname)s] [%(name)s] %(message)s"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tool to generate text data for experiments.",
        epilog="--input is not always required (for example, for random-letters.). However, --output is always required. I put all of the data in data/<DATA-NAME>. For example, I used --output data/news for the news dataset.",
    )
    parser.add_argument(
        "dataset",
        choices=[
            "news",
            "twitter",
            "reddit",
            "wikipedia",
            "openwebtext",
            "random-sentences",
            "random-words",
            "random-letters",
            "random-bytes",
            "pubmed",
        ],
    )
    parser.add_argument(
        "--input", type=str, nargs="+", help="Path to input file or directory"
    )
    parser.add_argument(
        "--output", type=str, help="Output folder for processed files", required=True
    )
    parser.add_argument(
        "--verbose",
        help="Whether to provide verbose output.",
        action="store_true",
        default=False,
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed to use.")

    return parser.parse_args()


def require_input(args):
    if not args.input:
        print(
            f"You need to provide an input file/directory for dataset '{args.dataset}'"
        )
        sys.exit(1)


def main():
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=log_format)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)

    import transformers

    transformers.set_seed(args.seed)

    if args.dataset == "twitter":
        from . import twitter

        twitter.preprocess(args.output)
    elif args.dataset == "reddit":
        from . import reddit

        reddit.preprocess(args.output)
    elif args.dataset == "wikipedia":
        from . import wikipedia

        wikipedia.preprocess(args.output)
    elif args.dataset == "pubmed":
        from . import pubmed

        pubmed.preprocess(args.output)
    elif args.dataset == "news":
        from . import news

        news.preprocess(args.output)
    elif args.dataset == "openwebtext":
        from . import openwebtext

        openwebtext.preprocess(args.output)
    elif args.dataset == "random-sentences":
        from nltk.tokenize import sent_tokenize

        from . import random_sequences

        require_input(args)
        sample_space = random_sequences.load_from_files(args.input, sent_tokenize)
        random_sequences.make_data(args.output, sample_space)
    elif args.dataset == "random-words":
        from nltk.tokenize import word_tokenize

        from . import random_sequences

        sample_space = random_sequences.wikipedia_sampling(word_tokenize)
        random_sequences.make_data(args.output, sample_space)
    elif args.dataset == "random-letters":
        from . import random_sequences

        sample_space = string.ascii_lowercase
        random_sequences.make_data(
            args.output, sample_space, with_replacement=True, separator=""
        )
    elif args.dataset == "random-bytes":
        from . import random_sequences

        sample_space = {chr(i) for i in range(256)}
        random_sequences.make_data(
            args.output, sample_space, with_replacement=True, separator=""
        )
    else:
        raise ValueError(f"Dataset {args.dataset} is not a valid choice.")


if __name__ == "__main__":
    main()
