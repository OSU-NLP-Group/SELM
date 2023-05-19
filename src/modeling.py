"""
Module to load models, tokenizers, and configs from disk/network. The aim is to make it as easy as possible to load a model, regardless of config options.

Every model will have an underlying causal language model, and some models will also inlude the fastfood.IntrinsicDimension (if there is an intrinsic dimension argument).
"""
import logging
from typing import Callable, List, Tuple

import numpy as np
import preface
import torch
import transformers

import intrinsic

from . import config, intrinsic_utils, modeling_utils, relic_helpers, util

logger = logging.getLogger(__name__)


class LanguageModel(
    torch.nn.Module, modeling_utils.Saveable, modeling_utils.KnowsBatchSize
):
    def __init__(self, language_model):
        self.m = [language_model]

    @property
    def hidden(self):
        return self.m[0]

    def __call__(self, *args, **kwargs):
        return self.hidden(*args, **kwargs)

    def __getattr__(self, name):
        if hasattr(self.hidden, name):
            return getattr(self.hidden, name)

        return super().__getattr__(name)

    def save(self, path):
        self.hidden.save_pretrained(path)

    def batch_size(self, training_config: config.TrainingConfig) -> int:
        """
        Based on available memory, determine how big a batch we can support.
        """
        logger.info("TODO: Take experiment_config.model.context_window into account.")
        mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024

        if mb <= util.rtx2080ti:
            # max on rtx2080ti is 128
            return min(128, training_config.batch_size)
        elif util.rtx2080ti < mb <= util.v100:
            # max on v100 is 128
            return min(256, training_config.batch_size)
        else:
            assert mb > util.v100
            # deal with this when the time comes
            return training_config.batch_size


class LanguageModelingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, model: torch.nn.Module
    ) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return self._cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )


class MaskedCrossEntropyLoss(torch.nn.Module):
    """
    If the top ranked option is correct, do not apply any loss.
    """

    def __init__(self):
        super().__init__()
        self._cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

    def _lm_loss(self, shift_logits, shift_labels):
        return self._cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, model: torch.nn.Module
    ) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        predictions = torch.argmax(shift_logits, dim=-1)

        shift_labels = labels[..., 1:].contiguous()
        mask = predictions != shift_labels

        losses = self._lm_loss(shift_logits, shift_labels)
        return torch.mean(losses.view(*mask.shape) * mask)


class UniformTokenProbabilityLoss(torch.nn.Module):
    n_classes: int
    target: float
    uniform: float

    def __init__(self, epsilon: float, n_classes: int):
        super().__init__()

        self._softmax = torch.nn.Softmax(dim=2)
        self._cross_entropy = torch.nn.CrossEntropyLoss()

        self.n_classes = n_classes
        self.target, self.uniform = self.compute_target_uniform(epsilon, n_classes)

    @staticmethod
    def compute_target_uniform(epsilon: float, n_classes: int) -> Tuple[float, float]:
        uniform = (1 - epsilon) / n_classes
        target = uniform + epsilon

        assert np.isclose(uniform * (n_classes - 1) + target, 1.0)

        return target, uniform

    def make_target_probalities(self, labels: torch.Tensor) -> torch.Tensor:
        batch, n = labels.shape
        target_probabilities = torch.full((batch, n, self.n_classes), self.uniform).to(
            labels.device
        )

        for i in range(batch):
            target_probabilities[i, torch.arange(n), labels[i]] = self.target

        return target_probabilities

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, model: torch.nn.Module
    ) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_probs = self._softmax(shift_logits)
        target_probs = self.make_target_probalities(shift_labels)

        return self._cross_entropy(shift_probs, target_probs)


class L2WeightPenalty(torch.nn.Module):
    def __init__(self, loss, weight):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, logits, labels, model):
        norm = torch.linalg.norm(torch.cat([p.flatten() for p in model.parameters()]))
        loss = self.loss(logits, labels, model)
        return loss + self.weight * norm


class MaxL2WeightPenalty(torch.nn.Module):
    def __init__(self, loss, weight, max):
        super().__init__()
        self.max = max
        self.loss = loss
        self.weight = weight

    def forward(self, logits, labels, model):
        norm = torch.linalg.norm(torch.cat([p.flatten() for p in model.parameters()]))
        loss = self.loss(logits, labels, model)
        if norm > self.max:
            return loss + self.weight * norm
        else:
            return loss


class TargetL2Norm(torch.nn.Module):
    def __init__(self, loss, weight, target):
        super().__init__()
        self.target = target
        self.loss = loss
        self.weight = weight

    def forward(self, logits, labels, model):
        norm = torch.linalg.norm(torch.cat([p.flatten() for p in model.parameters()]))
        dist = torch.abs(norm - self.target)
        loss = self.loss(logits, labels, model)

        logger.debug(
            "[L2 norm: %.4g, dist: %.4g, weight: %.4g, language loss: %.4g]",
            norm,
            dist,
            self.weight,
            loss,
        )

        return loss + self.weight * dist


class TargetL2NormD(TargetL2Norm):
    def forward(self, logits, labels, model):
        dist = torch.abs(model.L2_delta_theta_D - self.target)
        loss = self.loss(logits, labels, model)

        logger.debug(
            "[L2 norm: %.4g, dist: %.4g, weight: %.4g, language loss: %.4g, norm loss: %.4g]",
            model.L2_delta_theta_D,
            dist,
            self.weight,
            loss,
            self.weight * dist,
        )

        return loss + self.weight * dist


class TargetNormalDistribution(torch.nn.Module):
    name = "normal distribution"

    def __init__(self, loss, weight, *, mean, std):
        super().__init__()

        # Original loss (language modeling, most likely)
        self.loss = loss

        # Weight of normal distribution-based loss
        self.weight = weight

        # Target normal distribution parameters
        self.dist = torch.distributions.Normal(mean, std)


class KolmogorovSmirnovLoss(TargetNormalDistribution):
    name = "kolmogorov-smirnov"

    def statistic(self, observations):
        assert isinstance(observations, torch.Tensor)

        assert len(observations.shape) == 1, observations.shape
        n = observations.shape[0]

        ordered, _ = torch.sort(observations)

        empirical_cdf = torch.arange(0, n, device=ordered.device) / n
        cdf = self.dist.cdf(ordered)

        return torch.max(torch.abs(empirical_cdf - cdf))

    def forward(self, logits, labels, model):
        loss = self.loss(logits, labels, model)

        observations = torch.cat([p.flatten() for p in model.parameters()])

        stat = self.statistic(observations)

        logger.info(
            "[dist. stat: %.4g, weight: %.4g, language loss: %.4g, dist. loss: %.4g]",
            stat,
            self.weight,
            loss,
            self.weight * stat,
        )

        relic_helpers.log_step(
            **{
                self.name: float(stat),
                "weight": float(self.weight),
                "language loss": float(loss),
                f"{self.name} loss": float(self.weight * stat),
            }
        )

        return loss + self.weight * stat


class DistributionDifferenceSum(KolmogorovSmirnovLoss):
    name = "distribution-difference"

    def statistic(self, observations):
        assert isinstance(observations, torch.Tensor)

        assert len(observations.shape) == 1, observations.shape
        n = observations.shape[0]

        ordered, _ = torch.sort(observations, stable=True)

        empirical_cdf = torch.arange(0, n, device=ordered.device) / n
        cdf = self.dist.cdf(ordered)

        return torch.sum(torch.abs(empirical_cdf - cdf))


class DistributionDifferenceSumSquared(KolmogorovSmirnovLoss):
    name = "distribution-difference-squared"

    def statistic(self, observations):
        assert isinstance(observations, torch.Tensor)

        assert len(observations.shape) == 1, observations.shape
        n = observations.shape[0]

        ordered, _ = torch.sort(observations, stable=True)

        empirical_cdf = torch.arange(0, n, device=ordered.device) / n
        cdf = self.dist.cdf(ordered)

        return torch.dot(empirical_cdf - cdf, empirical_cdf - cdf)


class DistributionDifferenceIntegral(KolmogorovSmirnovLoss):
    name = "distribution-difference-integral"

    def statistic(self, observations):
        assert isinstance(observations, torch.Tensor)

        assert len(observations.shape) == 1, observations.shape
        n = observations.shape[0]

        ordered, _ = torch.sort(observations, stable=True)

        empirical_cdf = torch.arange(0, n, device=ordered.device) / n
        cdf = self.dist.cdf(ordered)

        # Uses trapezoid rule to approximate the integral.
        return torch.trapezoid(torch.abs(empirical_cdf - cdf), ordered)


def generate_random_vector(
    randomvec_config: config.RandomVecConfig,
    shape: Tuple[int, ...],
) -> torch.Tensor:
    """
    Generate a random tensor with a defined shape. The random values are drawn from the distribution described in randomvec_config.
    """
    rng = np.random.default_rng()

    assert randomvec_config.distribution is not None

    if randomvec_config.distribution == "normal":
        arr = rng.normal(**randomvec_config.dist_kwargs, size=shape)
    elif randomvec_config.distribution == "uniform":
        arr = rng.uniform(**randomvec_config.dist_kwargs, size=shape)
    elif randomvec_config.distribution == "loguniform":
        raise NotImplementedError()
    else:
        preface.never(randomvec_config.distribution)

    return torch.tensor(arr)


def add_random_vector(
    model: torch.nn.Module, randomvec_config: config.RandomVecConfig
) -> torch.nn.Module:
    """
    Adds a random value to each parameter in model using the information in randomvec_config.

    Assumes all random vector entries are independently sampled. This means we can edit each parameter in a loop rather than generating one single random vector.

    Both modifies the original model and returns it.
    """
    with torch.no_grad():
        for param in model.parameters():
            param += generate_random_vector(randomvec_config, param.shape)  # type: ignore

    return model


def new_projection_factory(
    model_config: config.ModelConfig, seed: int
) -> Callable[[int, int], torch.nn.Module]:
    """
    Creates a new projection factory for a projection from int_dim -> D, where D is the number of parameters in the true, underlying language model (unknown to us).

    If the model_config.projection.layers is only an empty list, then we just use a default fastfood transform.
    """
    intrinsic.implementation.set_seed(seed)

    def scaled_fastfood_transform(d: int, D: int) -> intrinsic.ScaledFastfoodTransform:
        return intrinsic.ScaledFastfoodTransform(d, D, model_config.scaling_factor)

    fastfood_cls: Callable[
        [int, int], intrinsic.FastfoodTransform
    ] = intrinsic.FastfoodTransform
    if model_config.scaled:
        # We use a function because we need to include the scaling factor
        fastfood_cls = scaled_fastfood_transform
    elif model_config.normalized:
        fastfood_cls = intrinsic.NormalizedFastfoodTransform

    logger.debug("Using linear projection. [class: %s]", fastfood_cls)

    layers = model_config.projection.layers
    if not layers:
        layers = ["output"]

    def factory(int_dim: int, D: int):
        """
        The factory function that makes a projection from int_dim to D.
        """
        modules: List[torch.nn.Module] = []
        input_dim = int_dim
        last_linear = False

        for val in layers:
            if isinstance(val, int):
                assert not last_linear
                modules.append(fastfood_cls(input_dim, val))
                input_dim = val
            elif isinstance(val, str):
                if val == "sigmoid":
                    modules.append(torch.nn.Sigmoid())
                elif val == "tanh":
                    modules.append(torch.nn.Tanh())
                elif val == "cos":
                    modules.append(modeling_utils.Cos())
                elif val == "sine" or val == "sine(x)":
                    modules.append(modeling_utils.Sine())
                elif val == "layernorm":
                    modules.append(modeling_utils.LayerNorm())
                elif val == "groupnorm":
                    assert (
                        "groups" in model_config.projection.layer_kwargs
                    ), "Projection configuration needs a 'groups' arg for GroupNorm!"
                    modules.append(
                        modeling_utils.GroupNorm(
                            model_config.projection.layer_kwargs["groups"]
                        )
                    )
                elif val == "nonlinear-wht":
                    modules.append(modeling_utils.NonlinearWHT())
                elif val == "1/x":
                    modules.append(modeling_utils.InverseFn())
                elif val == "dropout":
                    modules.append(torch.nn.Dropout(p=model_config.int_dim_dropout))
                elif val == "output":
                    modules.append(fastfood_cls(input_dim, D))
                    last_linear = True
                else:
                    preface.never(val)
            else:
                preface.never(val)

        assert last_linear, "You need at least one linear layer!"
        projection = torch.nn.Sequential(*modules)
        return projection

    return factory


def new(model_config: config.ModelConfig, vocab: int, seed: int):
    lm_config = transformers.AutoConfig.from_pretrained(
        model_config.language_model_name_or_path, cache_dir=util.HUGGINGFACE_CACHE_DIR
    )

    # context window size
    lm_config.n_ctx = model_config.context_window

    # configure dropout
    lm_config.resid_pdrop = model_config.dropout
    lm_config.attn_pdrop = model_config.dropout
    lm_config.embd_pdrop = model_config.dropout

    # configure checkpointing
    lm_config.gradient_checkpointing = False
    lm_config.use_cache = False

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_config.language_model_name_or_path,
        config=lm_config,
        cache_dir=util.HUGGINGFACE_CACHE_DIR,
    )

    # If we don't want to use the pretrained weights, then we need to re-initialize weights randomly.
    if not model_config.pretrained:
        transformers.set_seed(seed)
        model.init_weights()

    model.resize_token_embeddings(vocab)

    if model_config.random_vector.distribution is not None:
        model = add_random_vector(model, model_config.random_vector)

    if model_config.intrinsic_dimension is None:
        model = LanguageModel(model)
    else:
        projection_factory = new_projection_factory(model_config, seed)
        model = intrinsic_utils.IntrinsicDimension(
            model,
            model_config.intrinsic_dimension,
            said=model_config.intrinsic_dimension_said,
            projection_factory=projection_factory,
            seed=seed,
        )
        assert isinstance(model, modeling_utils.IntrinsicDimension)

    return model
