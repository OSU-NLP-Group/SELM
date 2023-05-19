import datetime
import logging
import math
import statistics
from typing import Any, Callable, Iterator, List, Sequence, Tuple

import preface
import relic
import torch
import transformers

from . import (
    accelerate,
    config,
    evaluating,
    modeling,
    modeling_utils,
    relic_helpers,
    training_utils,
)

logger = logging.getLogger(__name__)


def get_batch_size(training_config: config.TrainingConfig, model, dataset) -> int:
    """
    Returns the largest possible batch size for a given number of chunks, while respecting the max batch size in the training config.
    """
    memory_limited_batch_size = 1
    if training_config.batch_size > 1:
        assert isinstance(model, modeling_utils.KnowsBatchSize)
        memory_limited_batch_size = model.batch_size(training_config)

    return min(memory_limited_batch_size, len(dataset))


def make_dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=transformers.default_data_collator,
        batch_size=batch_size,
    )


def _make_optimizer(model, training_config: config.TrainingConfig):
    cls: Any
    if training_config.optimizer == "AdamW":
        cls = transformers.AdamW
    elif training_config.optimizer == "AdaFactor":
        cls = transformers.Adafactor
    elif training_config.optimizer == "SGD":
        cls = torch.optim.SGD
    elif training_config.optimizer == "RAdam":
        cls = torch.optim.RAdam
    elif training_config.optimizer == "Adam":
        cls = torch.optim.Adam
    elif training_config.optimizer == "NAdam":
        cls = torch.optim.NAdam
    else:
        preface.never(training_config.optimizer)

    return cls(
        model.parameters(),
        lr=training_config.learning_rate,
        **training_config.optim_kwargs,
    )


class DummyLRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self, *args):
        pass


def _make_lr_scheduler(
    optimizer, train_dataloader, training_config: config.TrainingConfig
):
    steps_per_epoch = len(train_dataloader)

    if training_config.lr_scheduler_type is None:
        return DummyLRScheduler(optimizer)
    elif training_config.lr_scheduler_type == "linear":
        return transformers.get_scheduler(
            name=training_config.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_config.warmup_epochs * steps_per_epoch,
            num_training_steps=training_config.maximum_epochs * steps_per_epoch,
        )
    elif training_config.lr_scheduler_type == "linear-const":
        decay = None
        warmup = None

        if training_config.decay_epochs > 0:
            decay = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.5,
                total_iters=steps_per_epoch * training_config.decay_epochs,
            )

        if training_config.warmup_epochs > 0:
            warmup_steps = steps_per_epoch * training_config.warmup_epochs
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        if decay is None and warmup is None:
            return DummyLRScheduler(optimizer)
        elif decay is None and warmup is not None:
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup], milestones=[]
            )
        elif decay is not None and warmup is None:
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[decay], milestones=[]
            )
        elif decay is not None and warmup is not None:
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, decay], milestones=[warmup_steps]
            )
        else:
            preface.never(decay)
            preface.never(warmup)
    if training_config.lr_scheduler_type == "reduce-on-plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            factor=training_config.plateau_reduce_factor,
            patience=training_config.plateau_patience,
            min_lr=training_config.plateau_min_lr,
            verbose=True,
        )
    else:
        preface.never(training_config.lr_scheduler_type)


def _make_criterion(train_config: config.TrainingConfig, tokenizer):
    """
    Makes the loss criterion object.

    If there is regularization, adds L2 weight regularization.
    """
    loss: Any

    if train_config.loss == "cross-entropy":
        loss = modeling.LanguageModelingLoss()
    elif train_config.loss == "hinge-ranked":
        loss = modeling.MaskedCrossEntropyLoss()
    elif train_config.loss == "uniform-probabilities":
        assert (
            "epsilon" in train_config.loss_kwargs
        ), "Need an 'epsilon' to use UniformTokenProbabilityLoss!"
        epsilon = train_config.loss_kwargs["epsilon"]
        loss = modeling.UniformTokenProbabilityLoss(epsilon, len(tokenizer))
    else:
        preface.never(train_config.loss)

    if train_config.regularization.variety is None:
        return loss

    if train_config.regularization.variety == "l2-weight-penalty":
        return modeling.L2WeightPenalty(loss, train_config.regularization.weight)
    elif train_config.regularization.variety == "max-l2-weight-penalty":
        return modeling.MaxL2WeightPenalty(
            loss, train_config.regularization.weight, train_config.regularization.max
        )
    elif train_config.regularization.variety == "target-l2-norm":
        return modeling.TargetL2Norm(
            loss, train_config.regularization.weight, train_config.regularization.max
        )
    elif train_config.regularization.variety == "target-l2-norm-D":
        return modeling.TargetL2NormD(
            loss, train_config.regularization.weight, train_config.regularization.max
        )
    elif train_config.regularization.variety == "kolmogorov-smirnov":
        return modeling.KolmogorovSmirnovLoss(
            loss,
            train_config.regularization.weight,
            mean=train_config.regularization.mean,
            std=train_config.regularization.std,
        )
    elif train_config.regularization.variety == "distribution-difference":
        return modeling.DistributionDifferenceSum(
            loss,
            train_config.regularization.weight,
            mean=train_config.regularization.mean,
            std=train_config.regularization.std,
        )
    elif train_config.regularization.variety == "distribution-difference-squared":
        return modeling.DistributionDifferenceSumSquared(
            loss,
            train_config.regularization.weight,
            mean=train_config.regularization.mean,
            std=train_config.regularization.std,
        )
    elif train_config.regularization.variety == "distribution-difference-integral":
        return modeling.DistributionDifferenceIntegral(
            loss,
            train_config.regularization.weight,
            mean=train_config.regularization.mean,
            std=train_config.regularization.std,
        )
    elif (
        train_config.regularization.variety == "unit-l2-norm"
        or train_config.regularization.variety == "max-l2-norm"
    ):
        # L2 norm regularization is applied as a weight-clipping method after optimization.
        return loss
    else:
        preface.never(train_config.regularization.variety)


def _accumulation_steps(
    desired_batch_size: int, limited_batch_size: int, example_count: int
) -> Tuple[int, ...]:
    """
    Calculates the epochs at which to step the optimizer for handling num_batches batches with a limited batch size (either by memory, or number of examples).

    Arguments:
        desired_batch_size (int): How many examples we'd like to see before stepping.
        limited_batch_size (int): The number of examples we can fit onto the GPU(s) for one forward/backward pass.
        example_count (int): The number of examples in the epoch.

    Returns:
        (int, int, ...): number of batches at which to step the optimizer at. The last value will always be the length of the dataloader.

    For example:
        Suppose your GPU constrained batch size is 2, you have 7 examples, and your desired batch size is 8. Then you should step after 7 examples, or 4 batches -> (4,)

    As another example:
        Suppose your GPU constrained batch size is 4, you have 11 examples and your desired batch size is 4. Then you should step after 4, 8, 11 examples, or 1, 2, 3 batches -> (1, 2, 3)
    """
    assert desired_batch_size > 0
    assert limited_batch_size > 0
    assert example_count > 0

    batch_count = math.ceil(example_count / limited_batch_size)

    steps: Sequence[int]

    # We don't need any accumulation if we can fit the batch size in memory.
    # So we step after every batch.
    if desired_batch_size <= limited_batch_size:
        steps = range(1, batch_count + 1)

    # If we have fewer examples in an epoch than the desired batch size, then just step at the end of every epoch.
    elif example_count < desired_batch_size:
        steps = (batch_count,)

    # At this point, we cannot fit the desired batch size into the GPU.
    # We have more examples per epoch than the desired batch size.
    # We need to step once after desired batch size examples.

    # This is the number of batches that need to a forward pass in order to accumulate enough gradients to reach desired batch size.
    # For example, if you want a batch size of 10 and your GPU can only do a batch size of 2, you need to do a forward pass on 5 batches to simulate a batch size of 10.
    else:
        assert desired_batch_size % limited_batch_size == 0
        required_batch_count = int(desired_batch_size / limited_batch_size)

        steps = list(range(required_batch_count, batch_count, required_batch_count))
        steps += [batch_count]

    steps = tuple(steps)
    assert steps[-1] == batch_count
    return steps


def _accumulation_factors(accumulation_steps: Sequence[int]) -> Iterator[int]:
    """
    Returns an iterator with length equal to the number of steps in the accumulation steps (so the last value in accumulation_steps). At each iteration, the iterator returns an int that describes how much to divide the loss by.
    """

    a = 0
    start = 0
    while a < len(accumulation_steps):
        end = accumulation_steps[a]
        factors = end - start

        yield from [factors] * factors

        a += 1
        start = end


def _should_accumulate(step: int, accumulation_steps: Sequence[int]) -> bool:
    return step + 1 in accumulation_steps


def _should_snapshot(epoch: int, experiment_config: config.ExperimentConfig) -> bool:
    if not experiment_config.training.snapshot_interval:
        return False

    if not experiment_config.model.intrinsic_dimension:
        return False

    return epoch % experiment_config.training.snapshot_interval == 0


def _calculate_epoch_pairs(
    training_config: config.TrainingConfig,
) -> List[Tuple[int, int]]:
    start_epochs = list(
        range(0, training_config.maximum_epochs, training_config.report_interval)
    )
    end_epochs = list(
        range(
            training_config.report_interval,
            training_config.maximum_epochs,
            training_config.report_interval,
        )
    ) + [training_config.maximum_epochs]

    return list(zip(start_epochs, end_epochs))


def _should_regularize(reg: config.RegularizationConfig, epoch: int) -> bool:
    if reg.variety is None:
        return False

    if reg.every == 0:
        return False

    return epoch % reg.every == 0


def _regularized(reg: config.RegularizationConfig, vec: torch.Tensor) -> torch.Tensor:
    if reg.variety is None:
        return vec

    if reg.variety == "unit-l2-norm":
        return vec / torch.linalg.vector_norm(vec, ord=2)
    elif reg.variety == "max-l2-norm":
        l2 = torch.linalg.vector_norm(vec, ord=2)
        if l2 > reg.max:
            return vec / l2 * reg.max
        else:
            return vec
    elif (
        reg.variety == "l2-weight-penalty"
        or reg.variety == "max-l2-weight-penalty"
        or reg.variety == "target-l2-norm"
        or reg.variety == "target-l2-norm-D"
        or reg.variety == "kolmogorov-smirnov"
        or reg.variety == "distribution-difference"
        or reg.variety == "distribution-difference-squared"
        or reg.variety == "distribution-difference-integral"
    ):
        # Do nothing because the weight-penalty is applied to the loss
        return vec
    else:
        preface.never(reg.variety)


class RegularizationWeightScheduler:
    def __init__(self, weight_fn: Callable[[int], float], reg_term):
        assert hasattr(
            reg_term, "weight"
        ), f"Regularization term {reg_term} must have a 'weight' attribute!"

        assert callable(weight_fn), f"Weight function {weight_fn} must be callable!"

        self.reg_term = reg_term
        self.initial_weight = self.reg_term.weight
        self.weight_fn = weight_fn

        self.reg_term.weight = 0

        self._step: int = 0

    def step(self) -> None:
        self._step += 1

        factor = self.weight_fn(self._step)
        self.reg_term.weight = self.initial_weight * factor


def _dummy_reg_scheduler(step: int):
    return 1.0


def _make_reg_scheduler(
    reg_config: config.RegularizationConfig, loss_criterion
) -> RegularizationWeightScheduler:
    if reg_config.schedule is None:
        fn = _dummy_reg_scheduler
    elif reg_config.schedule == "linear":
        fn = training_utils.make_linear_reg_scheduler(reg_config.warmup)
    else:
        preface.never(reg_config.schedule)

    return RegularizationWeightScheduler(fn, loss_criterion)


class Clipper:
    """
    Object that verifies that the clipping_config is correct in the constructor. Then we don't need any runtime asserts during the main loop.
    """

    def __init__(self, clipping_config: config.ClippingConfig):
        assert clipping_config.value > 0
        self.value = clipping_config.value

        assert clipping_config.algorithm is not None
        self.algorithm: config.ClippingAlgorithm = clipping_config.algorithm

    def _norm(self, parameters) -> float:
        with torch.no_grad():
            return torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2
            ).item()

    def clip(self, parameters) -> None:
        logger.debug(
            "Clipping parameters. [algorithm: %s, value: %.3g]",
            self.algorithm,
            self.value,
        )

        if self.algorithm == "value":
            torch.nn.utils.clip_grad_value_(parameters, self.value)
        elif self.algorithm == "norm":
            parameters = [p for p in parameters if p.grad is not None]
            relic_helpers.log_step(raw_gradient=self._norm(parameters))
            torch.nn.utils.clip_grad_norm_(parameters, self.value)
            relic_helpers.log_step(clipped_gradient=self._norm(parameters))
        else:
            preface.never(self.algorithm)


def log_parameter_metrics(epoch, model, loss) -> None:
    for param in model.parameters():
        logger.info(
            "[epoch: %s, loss: %.4g, vec L2 norm: %.4g, max grad: %.4g, min grad: %.4g, grad L2 norm: %.4g]",
            epoch,
            loss.item(),
            torch.linalg.norm(param),
            torch.max(param.grad),
            torch.min(param.grad),
            torch.linalg.norm(param.grad),
        )


def calc_delta_theta_l2_norm(model):
    norm = torch.linalg.norm(torch.cat([p.flatten() for p in model.parameters()]))
    return norm


def train(
    model,
    dataset,
    tokenizer,
    experiment_config: config.ExperimentConfig,
    experiment: relic.Experiment=None,
) -> Tuple[relic.Trial, Any]:
    """
    Fits `model` to `dataset` until it memorizes all values in `dataset`.

    Adds results to `experiment` during training in case training doesn't finish in time.
    """
    if experiment is None:
        relic_helpers.disable()
    else:
        trial = experiment.add_trial({})
        relic_helpers.register_trial(trial, experiment)

    relic_helpers.log_metric(finished=False, succeeded=False, epochs=0)
    relic_helpers.commit()

    batch_size = get_batch_size(experiment_config.training, model, dataset)
    dataloader = make_dataloader(dataset, batch_size)
    optimizer = _make_optimizer(model, experiment_config.training)
    lr_scheduler = _make_lr_scheduler(optimizer, dataloader, experiment_config.training)
    loss_criterion = _make_criterion(experiment_config.training, tokenizer)
    reg_scheduler = None
    if experiment_config.training.regularization.variety is not None:
        reg_scheduler = _make_reg_scheduler(
            experiment_config.training.regularization, loss_criterion
        )

    clipper = None
    if experiment_config.training.clipping.algorithm is not None:
        clipper = Clipper(experiment_config.training.clipping)

    accumulation_steps = _accumulation_steps(
        experiment_config.training.batch_size, batch_size, len(dataset)
    )

    logger.info("Running training")
    logger.info("[num examples: %s, num batches: %s]", len(dataset), len(dataloader))
    logger.info("[maximum epochs: %s]", experiment_config.training.maximum_epochs)
    logger.info(
        "[accumulation steps: %s, batch size: %s, desired batch size: %s]",
        accumulation_steps,
        batch_size,
        experiment_config.training.batch_size,
    )

    relic_helpers.log_metric(
        examples=len(dataset),
        batches=len(dataloader),
        accumulation_steps=accumulation_steps,
        actual_batch_size=batch_size,
    )
    relic_helpers.commit()

    model, optimizer, dataloader = accelerate.prepare(model, optimizer, dataloader)

    seconds_per_epochs = []
    epoch = -1

    for start, end in _calculate_epoch_pairs(experiment_config.training):
        # Check if the model has memorized the values in dataset.
        model.eval()
        eval_metrics = evaluating.passes(model, tokenizer, experiment_config, epoch)
        if eval_metrics.passed:
            relic_helpers.log_metric(succeeded=True)
            break
        else:
            relic_helpers.log_step(first_failed_token=eval_metrics.first_failed_token)
            logger.info(eval_metrics)

        # Fit model for (end - start) epochs.
        model.train()
        start_time = datetime.datetime.now()

        for epoch in range(start, end):
            # Zero the gradient at the start of each epoch
            optimizer.zero_grad()

            # Get a fresh set of factors
            accumulation_factors = _accumulation_factors(accumulation_steps)

            for step, batch in enumerate(dataloader):
                labels = batch.pop("labels")

                # Calculate loss
                outputs = model(**batch)
                acc_factor = next(accumulation_factors)
                loss = loss_criterion(outputs.logits, labels, model) / acc_factor
                loss.backward()

                relic_helpers.log_step(training_losses=float(loss))
                with torch.no_grad():
                    l2_norm = float(calc_delta_theta_l2_norm(model))
                    relic_helpers.log_step(delta_theta_l2_norms=l2_norm)

                if clipper is not None:
                    clipper.clip(model.parameters())

                log_parameter_metrics(epoch, model, loss)

                if _should_accumulate(step, accumulation_steps):
                    logger.debug(
                        "Applying accumulated gradients. [step: %s, accumulation steps: %s, last acc factor: %s]",
                        step,
                        accumulation_steps,
                        acc_factor,
                    )
                    optimizer.step()
                    if (
                        experiment_config.training.lr_scheduler_type
                        == "reduce-on-plateau"
                    ):
                        lr_scheduler.step(metrics=loss.item(), epoch=epoch)
                    else:
                        lr_scheduler.step()

                    if reg_scheduler is not None:
                        reg_scheduler.step()
                    optimizer.zero_grad()
                    relic_helpers.log_step(
                        learning_rates=optimizer.param_groups[0]["lr"]
                    )

            if _should_snapshot(epoch, experiment_config):
                assert isinstance(model, modeling_utils.IntrinsicDimension)
                relic_helpers.log_step(
                    intrinsic_dimension_snapshots=model.get_intrinsic_dimension_vector.view(
                        -1, 1
                    )
                )

            # Regularization step
            if _should_regularize(experiment_config.training.regularization, epoch):
                assert isinstance(model, modeling_utils.IntrinsicDimension)
                with torch.no_grad():
                    model.set_intrinsic_dimension_vector(
                        _regularized(
                            experiment_config.training.regularization,
                            model.get_intrinsic_dimension_vector,
                        )
                    )

        end_time = datetime.datetime.now()

        # training loss
        logger.info(
            f"Epoch {epoch + 1} had a training loss of {trial['training_losses'][-1]:.4f}."
        )

        # training speed
        iteration_count = end - start
        delta = end_time - start_time
        logger.info(
            "%s epochs took %s secs. Each iteration took %s secs.",
            iteration_count,
            delta,
            delta / iteration_count,
        )
        seconds_per_epochs.append((delta / iteration_count).total_seconds())

        relic_helpers.log_metric(
            seconds_per_epoch=statistics.mean(seconds_per_epochs),
            epochs=epoch + 1,
        )

        relic_helpers.commit()

    relic_helpers.log_metric(finished=True)
    relic_helpers.commit()

    return trial, model
