import collections
import statistics

from .. import relic_helpers


def load_experiments():
    filters = [
        "(~ data.file '300-tokens')",
        "(== model.intrinsic_dimension 3000)",
        "(== model.language_model_name_or_path 'gpt2')",
        "(not (== data.prompt_length None))",
        "(not (== data.prompt_type 'n-tokens'))",
        "(not (== data.prompt_type 'chunk-n'))",
    ]

    return relic_helpers.load_experiments(filters, show_cmd=True)


def print_table(experiments):
    # Dict[prefix type, int]
    counts = collections.defaultdict(list)

    for experiment in experiments:
        prefix = experiment.config["data"]["prompt_type"]

        counts[prefix].append(experiment[0]["epochs"])

    results = {}
    for prefix, epochs in counts.items():
        results[prefix] = (statistics.mean(epochs), statistics.stdev(epochs))

    mean, std = results["token"]
    print(f"New Token & \\num{{1}} & ${mean:.0f}\pm{std:.1f}$ \\\\")
    mean, std = results["vocab"]
    print(f"Vocab & \\num{{1}} & ${mean:.0f}\pm{std:.1f}$ \\\\")
    mean, std = results["natural-n"]
    print(f"Natural Prompt & \\num{{4}} & ${mean:.0f}\pm{std:.1f}$ \\\\")
    mean, std = results["uuid"]
    print(f"UUID & \\num{{27}} & ${mean:.0f}\pm{std:.1f}$ \\\\")
    mean, std = results["2x-uuid"]
    print(f"$2\\times$ UUID & \\num{{54}} & ${mean:.0f}\pm{std:.1f}$ \\\\")
    mean, std = results["3x-uuid"]
    print(f"$3\\times$ UUID & \\num{{76}} & ${mean:.0f}\pm{std:.1f}$ \\\\")


def main():
    experiments = load_experiments()

    print_table(experiments)


if __name__ == "__main__":
    main()
