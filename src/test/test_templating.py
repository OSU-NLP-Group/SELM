import numpy as np
import scipy.stats

from .. import templating


def test_expand_numbers_empty():
    s = ""

    expected = [""]
    actual = templating.expand_numbers(s)

    assert actual == expected


def test_expand_numbers_no_pattern():
    s = "data/file.txt"

    expected = [s]
    actual = templating.expand_numbers(s)

    assert actual == expected


def test_expand_numbers_no_pattern_with_digits():
    s = "data/file1.txt"

    expected = [s]
    actual = templating.expand_numbers(s)

    assert actual == expected


def test_expand_numbers_pattern_single_digits():
    s = "(0...10)"

    expected = list(range(0, 10))
    actual = templating.expand_numbers(s)

    assert actual == expected


def test_expand_numbers_pattern_single_digits_with_surrounding():
    s = "data/(0...10).txt"

    expected = [f"data/{i}.txt" for i in range(0, 10)]
    actual = templating.expand_numbers(s)

    assert actual == expected


def test_expand_numbers_pattern_big_numbers_with_surrounding():
    s = "data/(9876543000...9876543210).txt"

    expected = [f"data/{i}.txt" for i in range(9876543000, 9876543210)]
    actual = templating.expand_numbers(s)

    assert actual == expected


def test_expand_numbers_pattern_big_numbers():
    s = "(18446744073709551615...18446744073709551515)"

    expected = list(range(18446744073709551615, 18446744073709551515, -1))
    actual = templating.expand_numbers(s)

    assert actual == expected


def test_find_holes_empty():
    template = {}

    expected = []
    actual = templating.find_holes(template)

    assert actual == expected


def test_find_holes_no_holes():
    template = {"a": "b"}

    expected = []
    actual = templating.find_holes(template)

    assert actual == expected


def test_find_one_hole():
    template = {"a": ["b", "c"]}

    expected = [templating.Hole("a", templating.DiscreteDistribution(["b", "c"]))]
    actual = templating.find_holes(template)

    assert actual == expected


def test_find_two_holes():
    template = {"a": ["b", "c"], "d": ["e", "f"], "g": "h"}

    expected = [
        templating.Hole("a", templating.DiscreteDistribution(["b", "c"])),
        templating.Hole("d", templating.DiscreteDistribution(["e", "f"])),
    ]
    actual = templating.find_holes(template)

    assert actual == expected


def test_find_one_nested_hole():
    template = {
        "a": {"b": ["c", "d"]},
        "i": "j",
    }

    expected = [templating.Hole("a.b", templating.DiscreteDistribution(["c", "d"]))]
    actual = templating.find_holes(template)

    assert actual == expected


def test_find_two_nested_holes():
    template = {
        "a": {"b": ["c", "d"]},
        "e": {"f": ["g", "h"]},
        "i": "j",
    }

    expected = [
        templating.Hole("a.b", templating.DiscreteDistribution(["c", "d"])),
        templating.Hole("e.f", templating.DiscreteDistribution(["g", "h"])),
    ]
    actual = templating.find_holes(template)

    assert actual == expected


def test_find_holes_with_big_nums():
    template = {"seed": "(18446744073709551615...18446744073709551515)"}

    expected = [
        templating.Hole(
            "seed",
            templating.DiscreteDistribution(
                list(range(18446744073709551615, 18446744073709551515, -1))
            ),
        )
    ]
    actual = templating.find_holes(template)

    assert actual == expected


def test_find_multiple_holes_in_array():
    template = {
        "data": {
            "file": [
                "data/random-letters/1-chunks/(0...10).txt",
                "data/random-letters/2-chunks/(0...10).txt",
            ],
        },
        "model": {
            "intrinsic_dimension": [100000, 10000],
        },
    }

    expected = [
        templating.Hole(
            "data.file",
            templating.DiscreteDistribution(
                [f"data/random-letters/1-chunks/{i}.txt" for i in range(10)]
                + [f"data/random-letters/2-chunks/{i}.txt" for i in range(10)]
            ),
        ),
        templating.Hole(
            "model.intrinsic_dimension",
            templating.DiscreteDistribution([100000, 10000]),
        ),
    ]
    actual = templating.find_holes(template)

    assert actual == expected


def test_find_random_sample_distribution_holes():
    template = {
        "training": {
            "learning_rate__random-sample-distribution": {
                "dist": "uniform",
                "params": (5, 10),
            }
        }
    }

    expected = [
        templating.Hole(
            "training.learning_rate",
            templating.ContinuousDistribution(
                scipy.stats.uniform, {"loc": 5, "scale": 5}
            ),
        )
    ]

    actual = templating.find_holes(template)

    assert actual == expected


def test_find_normal_random_sample_distribution_holes():
    template = {
        "training": {
            "learning_rate__random-sample-distribution": {
                "dist": "normal",
                "params": (0, 1.5e-7),
            }
        }
    }

    expected = [
        templating.Hole(
            "training.learning_rate",
            templating.ContinuousDistribution(
                scipy.stats.norm, {"loc": 0, "scale": 1.5e-7}
            ),
        )
    ]

    actual = templating.find_holes(template)

    assert actual == expected


def test_find_loguniform_random_sample_distribution_holes():
    template = {
        "training": {
            "learning_rate__random-sample-distribution": {
                "dist": "loguniform",
                "params": (1e-9, 1e-7),
            }
        }
    }

    expected = [
        templating.Hole(
            "training.learning_rate",
            templating.ContinuousDistribution(
                scipy.stats.loguniform, {"a": 1e-9, "b": 1e-7}
            ),
        )
    ]

    actual = templating.find_holes(template)

    assert actual == expected


def test_no_expand_data_file():
    template = {
        "data": {
            "file": [
                "data/random-letters/1-chunks/(0...10).txt",
                "data/random-letters/2-chunks/(0...10).txt",
            ],
        },
        "model": {
            "intrinsic_dimension": [100000, 10000],
        },
    }

    expected = [
        {
            "data": {
                "file": [
                    "data/random-letters/1-chunks/(0...10).txt",
                    "data/random-letters/2-chunks/(0...10).txt",
                ],
            },
            "model": {
                "intrinsic_dimension": 100000,
            },
        },
        {
            "data": {
                "file": [
                    "data/random-letters/1-chunks/(0...10).txt",
                    "data/random-letters/2-chunks/(0...10).txt",
                ],
            },
            "model": {
                "intrinsic_dimension": 10000,
            },
        },
    ]
    actual = templating.generate(
        template, templating.Strategy.grid, no_expand={"data.file"}
    )

    assert actual == expected


def test_grid_fill_no_holes():
    primitives = {"a": 1}
    holes = []

    expected = [primitives]
    actual = templating.grid_fill(primitives, holes)

    assert actual == expected


def test_grid_fill_one_hole():
    primitives = {"a": 1}
    holes = [templating.Hole("b", templating.DiscreteDistribution([2, 3]))]

    expected = [{"a": 1, "b": 2}, {"a": 1, "b": 3}]
    actual = templating.grid_fill(primitives, holes)

    assert actual == expected


def test_grid_fill_one_nested_hole():
    primitives = {"a": 1}
    holes = [templating.Hole("b.c", templating.DiscreteDistribution([2, 3]))]

    expected = [
        {"a": 1, "b": {"c": 2}},
        {"a": 1, "b": {"c": 3}},
    ]
    actual = templating.grid_fill(primitives, holes)

    assert actual == expected


def test_grid_fill_two_holes():
    primitives = {"a": 1}
    holes = [
        templating.Hole("c", templating.DiscreteDistribution([4, 5])),
        templating.Hole("b", templating.DiscreteDistribution([2, 3])),
    ]

    expected = [
        {"a": 1, "b": 2, "c": 4},
        {"a": 1, "b": 2, "c": 5},
        {"a": 1, "b": 3, "c": 4},
        {"a": 1, "b": 3, "c": 5},
    ]
    actual = templating.grid_fill(primitives, holes)

    assert actual == expected


def test_random_fill_one_hole():
    holes = [
        templating.Hole(
            "b",
            templating.ContinuousDistribution(
                scipy.stats.uniform, {"loc": 0, "scale": 2}
            ),
        )
    ]

    actual = templating.random_fill(holes, 20)

    assert len(actual) == 20
    assert all(e["b"] >= 0 for e in actual)
    assert all(e["b"] <= 2 for e in actual)


def test_random_fill_normal_hole():
    holes = [
        templating.Hole(
            "b",
            templating.ContinuousDistribution(
                scipy.stats.norm, {"loc": 2, "scale": 0.001}
            ),
        )
    ]

    actual = templating.random_fill(holes, 1000)

    assert len(actual) == 1000
    np.testing.assert_allclose(np.mean([e["b"] for e in actual]), 2, atol=1e-4)


def test_generate_empty():
    template = {}

    expected = [template]
    actual = templating.generate(template, templating.Strategy.paired)

    assert actual == expected


def test_generate_no_holes():
    template = {"hello": "world"}

    expected = [template]

    actual = templating.generate(template, templating.Strategy.paired)
    assert actual == expected

    actual = templating.generate(template, templating.Strategy.grid)
    assert actual == expected


def test_generate_nested_fields():
    template = {"hello": {"world": "again"}}

    expected = [template]
    actual = templating.generate(template, templating.Strategy.paired)
    assert actual == expected

    actual = templating.generate(template, templating.Strategy.grid)
    assert actual == expected


def test_generate_one_list():
    template = {"hello": ["world", "universe"]}

    expected = [{"hello": "universe"}, {"hello": "world"}]

    actual = templating.generate(template, templating.Strategy.paired)
    assert actual == expected

    actual = templating.generate(template, templating.Strategy.grid)
    assert actual == expected


def test_generate_paired():
    template = {"hello": ["world", "universe"], "a": ["b", "c"], "d": 0}

    expected = [
        {"hello": "world", "a": "b", "d": 0},
        {"hello": "universe", "a": "c", "d": 0},
    ]
    actual = templating.generate(template, templating.Strategy.paired)

    assert actual == expected


def test_generate_with_one_pattern():
    template = {"a": "(0...5)"}

    expected = [{"a": i} for i in range(0, 5)]

    actual = templating.generate(template, templating.Strategy.grid)
    assert actual == expected

    actual = templating.generate(template, templating.Strategy.paired)
    assert actual == expected


def test_generate_with_two_patterns_paired():
    template = {"a": "(0...5)", "b": "data/file/(0...5).txt"}

    expected = [{"a": i, "b": f"data/file/{i}.txt"} for i in range(0, 5)]
    actual = templating.generate(template, templating.Strategy.paired)

    assert actual == expected


def test_generate_with_nested_patterns_paired():
    template = {"a": {"b": "(0...5)"}, "c": "data/file/(0...5).txt"}

    expected = [{"a": {"b": i}, "c": f"data/file/{i}.txt"} for i in range(0, 5)]
    actual = templating.generate(template, templating.Strategy.paired)

    assert actual == expected


def test_generate_grid():
    template = {"hello": ["world", "universe"], "a": ["b", "c"]}

    expected = [
        {"hello": "universe", "a": "b"},
        {"hello": "world", "a": "b"},
        {"hello": "universe", "a": "c"},
        {"hello": "world", "a": "c"},
    ]
    actual = templating.generate(template, templating.Strategy.grid)

    assert actual == expected


def test_generate_big_numbers():
    template = {"seed": "(18446744073709551615...18446744073709551515)"}

    expected = templating.sort_by_json(
        [{"seed": i} for i in range(18446744073709551615, 18446744073709551515, -1)]
    )

    actual = templating.generate(template, templating.Strategy.paired)
    assert actual == expected

    actual = templating.generate(template, templating.Strategy.grid)
    assert actual == expected


def test_nested_with_true_false():
    template = {"training": {"optim": {"nesterov": [True, False]}}}

    expected = templating.sort_by_json(
        [{"training": {"optim": {"nesterov": b}}} for b in [False, True]]
    )

    actual = templating.generate(template, templating.Strategy.paired)
    assert actual == expected

    actual = templating.generate(template, templating.Strategy.grid)
    assert actual == expected


def test_nested_with_two_options():
    template = {
        "training": {"optim": {"nesterov": [True, False], "momentum": [0, 0.9]}}
    }

    expected = templating.sort_by_json(
        [
            {"training": {"optim": {"momentum": 0, "nesterov": False}}},
            {"training": {"optim": {"momentum": 0, "nesterov": True}}},
            {"training": {"optim": {"momentum": 0.9, "nesterov": False}}},
            {"training": {"optim": {"momentum": 0.9, "nesterov": True}}},
        ]
    )

    actual = templating.generate(template, templating.Strategy.grid)
    assert actual == expected


def test_multiple_number_holes_in_array():
    template = {
        "data": {
            "file": [
                "data/random-letters/1-chunks/(0...10).txt",
                "data/random-letters/2-chunks/(0...10).txt",
                "data/random-letters/4-chunks/(0...10).txt",
                "data/random-letters/8-chunks/(0...10).txt",
                "data/random-letters/200-tokens/(0...10).txt",
            ],
        },
        "model": {
            "intrinsic_dimension": [100000, 10000],
        },
    }

    actual = templating.generate(template, templating.Strategy.grid)
    assert len(actual) == (10 + 10 + 10 + 10 + 10) * 2


def test_removes_dist_keys():
    template = {
        "learning_rate__random-sample-distribution": {
            "dist": "uniform",
            "params": (5, 10),
        }
    }

    actual = templating.generate(template, templating.Strategy.random, count=10)
    assert len(actual) == 10

    assert all("__random" not in key for config in actual for key in config)
