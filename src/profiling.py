import argparse
import logging
import os

import line_profiler
import torch
import transformers
from tqdm.auto import trange

from . import accelerate, experiments, intrinsic_utils, modeling, tokenizing, training

logger = logging.getLogger("profiling")


def profile_line(callback, output_dir):
    profiler = line_profiler.LineProfiler()
    profiler.add_function(callback)

    # IntrinsicDimension methods
    profiler.add_function(intrinsic_utils.IntrinsicDimension.set_module_weights)
    profiler.add_function(intrinsic_utils.IntrinsicDimension.forward)

    profiler.add_function(intrinsic_utils.FastfoodTransform._fastfood_transform)
    profiler.add_function(intrinsic_utils.FastfoodTransform.forward)

    profiler.add_function(intrinsic_utils.WalshHadamard.forward)
    logger.info("Added functions to profiler.")

    logger.info("Starting profiling.")
    profiler.enable()

    callback()

    profiler.disable()
    logger.info("Finished profiling")

    profiler.dump_stats(os.path.join(output_dir), "lines.prof")
    logger.info("Wrote stats.")


def profile_stacks(callback, output_dir):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
            torch.profiler.ProfilerActivity.CPU,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        callback()

    prof.export_chrome_trace(os.path.join(output_dir, "trace.json"))
    prof.export_stacks(
        os.path.join(output_dir, "cuda_stacks.txt"), "self_cuda_time_total"
    )
    prof.export_stacks(
        os.path.join(output_dir, "cpu_stacks.txt"), "self_cpu_time_total"
    )

    avgs = prof.key_averages()

    with open(os.path.join(output_dir, "cpu_times.txt"), "w") as file:
        file.write(avgs.table(sort_by="cpu_time", row_limit=10))

    with open(os.path.join(output_dir, "cuda_times.txt"), "w") as file:
        file.write(avgs.table(sort_by="cuda_time", row_limit=10))


def main(args: argparse.Namespace):
    experiment_config = next(
        experiments.find_experiments("experiments/gpt2/examples/advil/profile.toml")
    )
    logger.info("Loaded experiment config.")
    tokenizer = tokenizing.new(experiment_config.model.language_model_name_or_path)
    dataset = tokenizing.make_dataset(experiment_config.data, tokenizer)

    model = modeling.new(experiment_config.model, vocab=len(tokenizer), seed=0)

    logger.info("Loaded model and tokenizer.")

    model.train()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=transformers.default_data_collator,
        batch_size=experiment_config.training.batch_size,
    )

    model, dataloader = accelerate.prepare(model, dataloader)
    loss_criterion = training._make_criterion(experiment_config.training, tokenizer)

    output_dir = os.path.join(args.dir, args.name.replace("/", "-").replace(".", "_"))
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Warming CUDA up")
    for _ in trange(10):
        for batch in dataloader:
            labels = batch.pop("labels")

            # calculate loss
            outputs = model(**batch)
            loss = loss_criterion(outputs.logits, labels)
            loss.backward()

    logger.info("Starting profiling.")
    if args.type == "line":
        profile_line(lambda: do_work(50), output_dir)
    elif args.type == "stacks":
        profile_line(lambda: do_work(1), output_dir)
    else:
        raise ValueError(f"Profile type '{type}' is not supported!")

    def do_work(epochs):
        for _ in trange(epochs):
            for batch in dataloader:
                labels = batch.pop("labels")

                # calculate loss
                outputs = model(**batch)
                loss = loss_criterion(outputs.logits, labels)
                loss.backward()


if __name__ == "__main__":
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "type", choices=["line", "stacks"], help="Type of profiling to use"
    )
    parser.add_argument("--dir", default="logs", help="output_directory")
    parser.add_argument("name", help="Name")
    args = parser.parse_args()
    main(args)
