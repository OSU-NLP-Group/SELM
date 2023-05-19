import torch

import intrinsic

from .. import config, modeling


def test_imports():
    pass


def check_fft_equality(a, b):
    return a.d == b.d and a.D == b.D


def assert_sequential(actual, expected):
    assert len(actual) == len(expected)

    for a, e in zip(actual, expected):
        if isinstance(e, intrinsic.FastfoodTransform):
            assert check_fft_equality(a, e)
        else:
            assert type(a) == type(e)


def test_project_factory_empty():
    model_config = config.ModelConfig(
        language_model_name_or_path="gpt2",
        projection=config.ProjectionConfig(layers=[]),
    )

    int_dim, D = 100, 1000

    factory = modeling.new_projection_factory(model_config, seed=0)
    projection = factory(int_dim, D)

    expected = torch.nn.Sequential(intrinsic.FastfoodTransform(int_dim, D))

    assert_sequential(projection, expected), str(projection)


def test_project_factory_nonlinearity():
    model_config = config.ModelConfig(
        language_model_name_or_path="gpt2",
        projection=config.ProjectionConfig(layers=["output", "sigmoid"]),
    )

    int_dim, D = 100, 1000

    factory = modeling.new_projection_factory(model_config, seed=0)
    projection = factory(int_dim, D)

    expected = torch.nn.Sequential(
        intrinsic.FastfoodTransform(int_dim, D), torch.nn.Sigmoid()
    )

    assert_sequential(projection, expected)


def test_project_factory_two_projection():
    model_config = config.ModelConfig(
        language_model_name_or_path="gpt2",
        projection=config.ProjectionConfig(layers=[500, "sigmoid", "output"]),
    )

    int_dim, D = 100, 1000

    factory = modeling.new_projection_factory(model_config, seed=0)
    projection = factory(int_dim, D)

    expected = torch.nn.Sequential(
        intrinsic.FastfoodTransform(int_dim, 500),
        torch.nn.Sigmoid(),
        intrinsic.FastfoodTransform(500, 1000),
    )

    assert_sequential(projection, expected)


def test_project_factory_neuralnetwork():
    model_config = config.ModelConfig(
        language_model_name_or_path="gpt2",
        projection=config.ProjectionConfig(
            layers=[500, "sigmoid", "output", "sigmoid"]
        ),
    )

    int_dim, D = 100, 1000

    factory = modeling.new_projection_factory(model_config, seed=0)
    projection = factory(int_dim, D)

    expected = torch.nn.Sequential(
        intrinsic.FastfoodTransform(int_dim, 500),
        torch.nn.Sigmoid(),
        intrinsic.FastfoodTransform(500, 1000),
        torch.nn.Sigmoid(),
    )

    assert_sequential(projection, expected)


def test_kolmogorov_smirnov_empirical_cdf_simple():
    ks = modeling.KolmogorovSmirnovLoss(None, None, mean=0, std=1)

    observations = torch.tensor([0, 0.3, 0.4, 0.8, 1.5])

    assert ks.statistic(observations) == 0.5
