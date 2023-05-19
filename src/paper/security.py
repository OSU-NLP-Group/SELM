"""
This script helps you fill in the security results table, cell by cell.
"""
import argparse
from typing import Callable, Dict, Optional

import numpy as np

from .. import attacking, logging, relic_helpers

logging.init(__name__, verbose=False, date=False)


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        choices=["lda", "svm", "knn", "gradboost", "ffnn"],
        help="Which attack model to use",
    )
    parser.add_argument(
        "group",
        choices=[
            "original",
            "l2-norm-reg",
            "distribution-reg",
        ],
        help="Which ciphertext groups to use",
    )
    parser.add_argument(
        "repr",
        choices=["cipher", "feature-fn"],
        help="Which ciphertext representation to use as input",
    )
    parser.add_argument("--seed", type=int, help="Random seed to use.", required=True)
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.8,
        help="Train-test split ratio.",
    )
    parser.add_argument(
        "--relics",
        help="Path to relics/ directory",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Show output in simplified format for copying to paper.",
    )
    parser.add_argument(
        "count", type=int, help="How many experiments from each group to use"
    )

    return parser


def load_ciphertexts(group: str, count: int, root=None) -> Dict[str, np.ndarray]:
    if group == "original":
        filters = [
            "(== training.regularization.variety None)",
            "(== seed_source 'random')",
            "(== model.language_model_name_or_path 'gpt2')",
            "(== model.intrinsic_dimension 10000)",
            "(== tokenizer.variety 'pretrained')",
            "(== training.batch_size 1)",
            "(== data.prompt_length None)",
            r"""(or
                    (~ data.file 'news/100-tokens/(0|1)')
                    (~ data.file 'pubmed/100-tokens/0')
                    (~ data.file 'random-words/100-tokens/0')
                    (~ data.file 'random-bytes/100-tokens/0'))""",
            "(== training.lr_scheduler_type 'linear-const')",
        ]
    elif group == "l2-norm-reg":
        filters = [
            "(== training.regularization.variety 'target-l2-norm')",
            "(== seed_source 'random')",
            "(== model.language_model_name_or_path 'gpt2')",
            "(== model.intrinsic_dimension 10000)",
            "(== tokenizer.variety 'pretrained')",
            r"""(or
                    (~ data.file 'news/100-tokens/(0|1)')
                    (~ data.file 'pubmed/100-tokens/0')
                    (~ data.file 'random-words/100-tokens/0')
                    (~ data.file 'random-bytes/100-tokens/0'))""",
            "(== training.lr_scheduler_type 'linear-const')",
        ]
    elif group == "distribution-reg":
        filters = [
            "(== training.regularization.variety 'distribution-difference-integral')",
            "(== seed_source 'random')",
            "(== model.language_model_name_or_path 'gpt2')",
            "(== model.intrinsic_dimension 10000)",
            "(== tokenizer.variety 'pretrained')",
            r"""(or
                    (~ data.file 'news/100-tokens/(0|1)')
                    (~ data.file 'pubmed/100-tokens/0')
                    (~ data.file 'random-words/100-tokens/0')
                    (~ data.file 'random-bytes/100-tokens/0'))""",
            r"(== training.regularization.weight 500000000)",
            "(== training.lr_scheduler_type 'linear-const')",
        ]
    else:
        raise ValueError(group)

    experiments = relic_helpers.load_experiments(filters, root=root, show_cmd=True)
    ciphertexts = attacking.data.load_ciphertexts(experiments, count)

    for file, stack in ciphertexts.items():
        assert stack.shape == (
            count,
            10_000,
        ), f"File {file} only has {stack.shape[0]} ciphertexts!"

    return ciphertexts


def load_feature_fn(repr: str) -> Optional[attacking.data.PreprocessFn]:
    if repr == "cipher":
        return None
    elif repr == "feature-fn":
        return attacking.data.preprocess
    else:
        raise ValueError(repr)


def load_model_fn(
    model: str, seed: int, feature_fn: Optional[attacking.data.PreprocessFn]
) -> Callable[[], attacking.semantic_security.Model]:
    if model == "lda":
        return attacking.lda.init_model
    if model == "svm":
        return attacking.svm.init_model
    if model == "knn":
        return attacking.knn.init_model
    if model == "gradboost":
        return lambda: attacking.gradboost.init_model(seed)
    if model == "ffnn":
        return lambda: attacking.ffnn.init_model(feature_fn)
    if model == "xgboost":
        return attacking.xgboost.init_model
    else:
        raise ValueError(model)


def main():
    parser = init_parser()

    args = parser.parse_args()

    # 1. Load feature_fn
    feature_fn = load_feature_fn(args.repr)

    # 2. Load ciphertexts
    ciphertexts = load_ciphertexts(args.group, args.count, args.relics)
    datasets = attacking.data.make_single_datasets(
        ciphertexts, args.seed, preprocess_fn=feature_fn, ratio=args.ratio
    )

    # 3. Load model_fn
    model_fn = load_model_fn(args.model, args.seed, feature_fn)

    # 4. Play the game
    passed = attacking.semantic_security.play(
        datasets, model_fn, args.model, args.seed, quiet=args.quiet
    )

    # 5. Report the results
    print(f"Passed: {passed}")


if __name__ == "__main__":
    main()
