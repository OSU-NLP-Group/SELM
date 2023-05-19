import argparse
import os
import pathlib

import tomli
import tomli_w
from tqdm.auto import tqdm

from .. import config, logging, templating, util

logger = logging.init("experiments.generate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="General .toml files from template .toml files. I kept all my templates in experiments/templates and my generated experiment configs in experiments/generated, which I then removed from version control.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="Strategy to use to combine multiple lists in a template.",
        default="grid",
        choices=["grid", "paired", "random"],
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Number of configs to generate when using --strategy random. Required.",
        default=-1,
    )
    parser.add_argument(
        "--no-expand",
        type=str,
        nargs="+",
        default=[],
        help=".-separated fields to not expand",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="generated-",
        help="Prefix to add to generated templates",
    )
    parser.add_argument(
        "templates",
        nargs="+",
        type=str,
        help="Template .toml files or directories containing template .toml files.",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output directory to write the generated .toml files to.",
    )
    return parser.parse_args()


def generate(args: argparse.Namespace) -> None:
    strategy = templating.Strategy.new(args.strategy)

    count = args.count
    if strategy is templating.Strategy.random:
        assert count > 0, "Need to include --count!"

    for template_toml in util.files_with_extension(args.templates, ".toml"):
        with open(template_toml, "rb") as template_file:
            try:
                template_dict = tomli.load(template_file)
            except tomli.TOMLDecodeError as err:
                logger.warning(
                    "Error parsing template file. [file: %s, err: %s]",
                    template_toml,
                    err,
                )
                continue

        template_name = pathlib.Path(template_toml).stem

        logger.info("Opened template file. [file: %s]", template_toml)

        experiment_dicts = templating.generate(
            template_dict, strategy, count=count, no_expand=set(args.no_expand)
        )

        logger.info(
            "Loaded experiment dictionaries. [count: %s]", len(experiment_dicts)
        )

        for i, experiment_dict in enumerate(tqdm(experiment_dicts)):
            filename = f"{args.prefix}{template_name}-{i}.toml"
            filepath = os.path.join(args.output, filename)
            with open(filepath, "wb") as file:
                tomli_w.dump(experiment_dict, file)

            # Verifies that the configs are correctly loaded.
            list(config.load_configs(filepath))


def main() -> None:
    args = parse_args()
    generate(args)


if __name__ == "__main__":
    main()
