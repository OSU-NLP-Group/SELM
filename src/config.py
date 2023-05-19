import copy
import dataclasses
import json
import logging
import os
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
)

import tomli

from . import util

T = TypeVar("T", bound="Config")

PromptType = Literal[
    "uuid", "token", "vocab", "chunk-n", "natural-n", "2x-uuid", "3x-uuid", "n-tokens"
]
Loss = Literal["cross-entropy", "hinge-ranked", "uniform-probabilities"]
SeedSource = Literal["trial", "config", "random"]
Optimizer = Literal["Adam", "AdamW", "RAdam", "NAdam", "AdaFactor", "SGD"]
Regularization = Literal[
    "unit-l2-norm",
    "max-l2-norm",
    "l2-weight-penalty",
    "max-l2-weight-penalty",
    "target-l2-norm",
    "target-l2-norm-D",
    "kolmogorov-smirnov",
    "distribution-difference",
    "distribution-difference-squared",
    "distribution-difference-integral",
]
Schedule = Literal["linear", "linear-const"]
LearningRateSchedule = Literal["linear", "linear-const", "reduce-on-plateau", "const"]
Distribution = Literal["normal", "uniform", "loguniform"]
Layer = Union[
    int,
    Literal[
        "sigmoid",
        "tanh",
        "output",
        "cos",
        "sine",
        "layernorm",
        "groupnorm",
        "1/x",
        "nonlinear-wht",
        "dropout",
    ],
]
ClippingAlgorithm = Literal["value", "norm"]
Tokenizer = Literal["pretrained", "1-byte", "2-byte"]

logger = logging.getLogger(__name__)


class Config:
    @classmethod
    def from_dict(cls: Type[T], dct: Dict[str, Any]) -> T:
        for field in dataclasses.fields(cls):
            if (
                isinstance(field.type, type)
                and issubclass(field.type, Config)
                and field.name in dct
                and not isinstance(dct[field.name], field.type)
            ):
                if not isinstance(dct[field.name], dict):
                    logger.warn(
                        "Subdict is not a dict! [cls: %s, field name: %s, field type: %s, actual type: %s]",
                        cls,
                        field.name,
                        field.type,
                        type(dct[field.name]),
                    )
                dct[field.name] = field.type.from_dict(dct[field.name])

        return cls(**dct)

    @classmethod
    def get_toml_name(cls) -> str:
        # Because I'm a bad programmer and I do hacky things.
        return cls.__name__[: cls.__name__.lower().find("config")].lower()

    @classmethod
    def from_existing(cls: Type[T], other: Type[T], **overrides) -> T:
        kwargs = {**dataclasses.asdict(other), **overrides}

        return cls(**kwargs)

    @property
    def pretty(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=4)

    def __str__(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    def validate_field(self, fname: str, ftype) -> None:
        choices = get_args(ftype)
        if getattr(self, fname) not in choices:
            raise ValueError(f"self.{fname} must be one of {', '.join(choices)}")


@dataclasses.dataclass(frozen=True)
class RandomVecConfig(Config):
    distribution: Optional[Distribution] = None

    # Distribution keyword args.
    dist_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.distribution is not None:
            self.validate_field("distribution", Distribution)


@dataclasses.dataclass(frozen=True)
class ProjectionConfig(Config):
    layers: List[Layer] = dataclasses.field(default_factory=lambda: ["output"])
    layer_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, dct) -> "ProjectionConfig":
        """
        I reimplement this method because the toml dict will have a string for layers that needs to be evaluated to a real Python list.
        """
        for key in dct:
            if key == "layers" and isinstance(dct[key], str):
                dct[key] = eval(dct[key])

        return cls(**dct)


@dataclasses.dataclass(frozen=True)
class ModelConfig(Config):
    language_model_name_or_path: str
    intrinsic_dimension: Optional[int] = None

    # Structure-aware intrinsic dimension (SAID)
    # Has no effect when intrinsic_dimension is None.
    intrinsic_dimension_said: bool = False

    # temperature of 1.0 has no effect, lower tend toward greedy sampling
    temperature: float = 1.0

    # The number of highest probability vocabulary tokens to keep for top-k-filtering.
    top_k: int = 0

    # If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    top_p: float = 0.9

    # primarily useful for CTRL model; in that case, use 1.2
    repetition_penalty: float = 1.0

    # optional stop token (ignore text generated after this token) [DEPRECATED]
    stop_token: Optional[str] = None

    # context window size
    context_window: int = 1024

    # dropout probability for fully connected layers in embeddings, encoder, and pooler, embeddings, and attention.
    dropout: float = 0.0

    # dropout probability for the intrinsic dimension layer(s)
    int_dim_dropout: float = 0.0

    # Whether to use pre-trained weights.
    pretrained: bool = True

    random_vector: RandomVecConfig = dataclasses.field(default_factory=RandomVecConfig)

    projection: ProjectionConfig = dataclasses.field(default_factory=ProjectionConfig)

    normalized: bool = True
    scaled: bool = False
    scaling_factor: float = 1

    def __post_init__(self) -> None:
        assert isinstance(self.random_vector, RandomVecConfig), str(
            type(self.random_vector)
        )
        assert isinstance(self.projection, ProjectionConfig), str(type(self.projection))


@dataclasses.dataclass(frozen=True)
class RegularizationConfig(Config):
    # Which kind of regularization to use.
    variety: Optional[Regularization] = None

    # How often to regularize
    every: int = 0

    # Maximum allowed vaue
    # For target-l2-norms, this is the target value.
    max: float = 0

    # Weight for the regularization term in loss
    weight: float = 0

    # Mean and std deviation for normal-distribution-based regularization
    mean: float = 0
    std: float = 0

    # How to schedule the value of the weight term
    schedule: Optional[Schedule] = None
    # Number of steps before weight term reaches maximum
    warmup: int = 0

    def __post_init__(self) -> None:
        if self.variety is not None:
            self.validate_field("variety", Regularization)

        if self.schedule is not None:
            self.validate_field("schedule", Schedule)


@dataclasses.dataclass(frozen=True)
class ClippingConfig(Config):
    value: float = 0
    algorithm: Optional[ClippingAlgorithm] = None

    def __post_init__(self) -> None:
        if self.algorithm is not None:
            self.validate_field("algorithm", ClippingAlgorithm)


@dataclasses.dataclass(frozen=True)
class TrainingConfig(Config):
    learning_rate: float = 5e-3
    batch_size: int = 1
    optimizer: Optimizer = "AdamW"

    # Optimizer keyword args.
    optim: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "betas": {"b1": 0.9, "b2": 0.999},
            "correct_bias": True,
        }
    )

    # Regularization options.
    regularization: RegularizationConfig = dataclasses.field(
        default_factory=RegularizationConfig
    )

    # Clipping options.
    clipping: ClippingConfig = dataclasses.field(default_factory=ClippingConfig)

    # How often to check if memorized.
    report_interval: int = 100

    # How often to take an intrinsic dimension snapshot.
    # 0 means to never take a snapshot.
    snapshot_interval: int = 0

    # Maximum number of epochs to train for.
    maximum_epochs: int = 20_000

    # Learning rate scheduling
    lr_scheduler_type: LearningRateSchedule = "const"
    # Number of warmup epochs for learning rates
    warmup_epochs: int = 0
    # Fixed number of epochs to decay over
    decay_epochs: int = 0
    # Proportion of initial learning rate to decay to.
    # Number between 0 and 1.
    decayed_proportion: float = 0.5

    # linear-const learning rate schedule
    # |    *  <- learning_rate
    # |   * *
    # |  *   *
    # | *     ********* <- learning_rate * decay_proportion
    # |*  <- learning_rate * 0.1
    # +----------------
    #  |---| <- warmup_epochs
    #      |--| <- decay_epochs

    # Plateau Configuration
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    # Factor by which the learning rate will be reduced.
    plateau_reduce_factor: float = 0.1
    # Number of epochs with no improvement after which learning rate will be reduced.
    plateau_patience: float = 10
    # A lower bound on the learning rate
    plateau_min_lr: float = 1e-12

    weight_decay: float = 0.0
    loss: Loss = "cross-entropy"
    loss_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate_field("loss", Loss)
        self.validate_field("optimizer", Optimizer)
        if self.lr_scheduler_type is not None:
            self.validate_field("lr_scheduler_type", LearningRateSchedule)

        assert isinstance(self.regularization, RegularizationConfig), str(
            type(self.regularization)
        )

        assert isinstance(self.clipping, ClippingConfig), str(type(self.clipping))
        assert isinstance(self.maximum_epochs, int)
        assert isinstance(self.warmup_epochs, int)
        assert isinstance(self.decay_epochs, int)

    @property
    def optim_kwargs(self) -> Dict[str, Any]:
        kwargs = {}

        # Handle anything not natively supported by TOML.
        betas = self.optim.pop("betas", None)
        if betas:
            kwargs["betas"] = (betas["b1"], betas["b2"])

        # Handle everything natively supported
        kwargs = {**kwargs, **self.optim}

        # Replace the stuff we popped
        self.optim["betas"] = betas

        return kwargs


@dataclasses.dataclass(frozen=True)
class DataConfig(Config):
    file: str
    overwrite_cache: bool = False
    """
    Possible Values
    * uuid: encodes a uuid as the prompt (typically between 20-30 tokens for GPT2).
    * 2x-uuid: 2 uuids
    * 3x-uuid: 3 uuids
    * token: adds a new token to the vocabulary for each chunk (<|start0|>, <|start1|>, etc.)
    * vocab: finds an existing token in the vocabulary that's not in any of the examples and uses it as th e prompt.
    * chunk-n: "Chunk 1: ", "Chunk 2: ", ...
    * natural-n: "The first chunk is: ", "The second chunk is: ", ...
    * n-tokens: n tokens randomly drawn from the token space
    """
    prompt_type: PromptType = "uuid"

    # Only used if prompt_type is 'n-tokens'
    prompt_length: int = 0

    chunk_length: Union[Literal["longest"], int] = "longest"

    def __post_init__(self) -> None:
        if not os.path.exists(self.file):
            raise ValueError(f"{self.file} does not exist!")

        self.validate_field("prompt_type", PromptType)

        if self.chunk_length != "longest":
            assert isinstance(self.chunk_length, int)

    def get_text(self) -> str:
        assert self.file is not None

        with open(self.file, "r") as file:
            return file.read()


@dataclasses.dataclass(frozen=True)
class ExperimentConfig(Config):
    model: ModelConfig
    # Describes which tokenizer to use.
    # * pretrained is a pretrained tokenizer. It uses the model config's
    #   language_model_name_or_path field.
    # * 1-byte means 256 tokens (1 token for all possible bytes)
    # * 2-byte means 256 * 256 tokens (1 token for all possible 2-byte sequences).
    tokenizer: Tokenizer
    data: DataConfig
    training: TrainingConfig

    trials: int = 3
    save_weights: bool = True
    seed_source: SeedSource = "trial"
    seed: int = 0

    def __post_init__(self) -> None:
        self.validate_field("seed_source", SeedSource)


def load_configs(config_file: str) -> Iterator[ExperimentConfig]:
    """
    A config file could contain many experiments. For any field in a config file, if it is a list, then it turns into multiple experiments. If there are multiple lists, then each combination of elements from each list forms a new experiment.
    """
    with open(config_file, "r") as file:
        config_dict = tomli.loads(file.read())

    for flat in util.flattened(config_dict):
        yield ExperimentConfig.from_dict(copy.deepcopy(flat))
