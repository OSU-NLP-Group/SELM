from .. import training


def test_accumulation_steps1() -> None:
    desired_batch_size = 128
    memory_limited_batch_size = 2
    example_count = 16

    expected = (8,)

    actual = training._accumulation_steps(
        desired_batch_size, memory_limited_batch_size, example_count
    )

    assert actual == expected


def test_accumulation_steps2() -> None:
    desired_batch_size = 256
    memory_limited_batch_size = 2
    example_count = 16

    expected = (8,)

    actual = training._accumulation_steps(
        desired_batch_size, memory_limited_batch_size, example_count
    )

    assert actual == expected


def test_accumulation_steps3() -> None:
    desired_batch_size = 256
    memory_limited_batch_size = 4
    example_count = 7

    expected = (2,)

    actual = training._accumulation_steps(
        desired_batch_size, memory_limited_batch_size, example_count
    )

    assert actual == expected


def test_accumulation_steps4() -> None:
    desired_batch_size = 256
    memory_limited_batch_size = 4
    example_count = 1

    expected = (1,)

    actual = training._accumulation_steps(
        desired_batch_size, memory_limited_batch_size, example_count
    )

    assert actual == expected


def test_accumulation_steps5() -> None:
    desired_batch_size = 2
    memory_limited_batch_size = 2
    example_count = 32

    expected = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

    actual = training._accumulation_steps(
        desired_batch_size, memory_limited_batch_size, example_count
    )

    assert actual == expected


def test_accumulation_steps6() -> None:
    desired_batch_size = 4
    memory_limited_batch_size = 2
    example_count = 7

    expected = (2, 4)

    actual = training._accumulation_steps(
        desired_batch_size, memory_limited_batch_size, example_count
    )

    assert actual == expected


def test_accumulation_steps7() -> None:
    desired_batch_size = 8
    memory_limited_batch_size = 2
    example_count = 7

    expected = (4,)

    actual = training._accumulation_steps(
        desired_batch_size, memory_limited_batch_size, example_count
    )

    assert actual == expected


def test_accumulation_steps8() -> None:
    desired_batch_size = 8
    memory_limited_batch_size = 2
    example_count = 9

    expected = (4, 5)

    actual = training._accumulation_steps(
        desired_batch_size, memory_limited_batch_size, example_count
    )

    assert actual == expected


def test_accumulation_steps9() -> None:
    desired_batch_size = 16
    memory_limited_batch_size = 2
    example_count = 9

    expected = (5,)

    actual = training._accumulation_steps(
        desired_batch_size, memory_limited_batch_size, example_count
    )

    assert actual == expected


def test_accumulation_steps10() -> None:
    desired_batch_size = 512
    memory_limited_batch_size = 64
    example_count = 9

    expected = (1,)

    actual = training._accumulation_steps(
        desired_batch_size, memory_limited_batch_size, example_count
    )

    assert actual == expected


def test_accumulation_steps11() -> None:
    desired_batch_size = 16
    memory_limited_batch_size = 5
    example_count = 5

    expected = (1,)

    actual = training._accumulation_steps(
        desired_batch_size, memory_limited_batch_size, example_count
    )

    assert actual == expected


def test_accumulation_steps12() -> None:
    desired_batch_size = 4
    memory_limited_batch_size = 2
    example_count = 9

    expected = (2, 4, 5)

    actual = training._accumulation_steps(
        desired_batch_size, memory_limited_batch_size, example_count
    )

    assert actual == expected


def test_accumulation_steps13() -> None:
    desired_batch_size = 4
    memory_limited_batch_size = 2
    example_count = 8

    expected = (2, 4)

    actual = training._accumulation_steps(
        desired_batch_size, memory_limited_batch_size, example_count
    )

    assert actual == expected


def test_accumulation_factors1() -> None:
    accumulation_steps = (8, 11)
    expected = [8, 8, 8, 8, 8, 8, 8, 8, 3, 3, 3]
    actual = list(training._accumulation_factors(accumulation_steps))

    assert actual == expected


def test_accumulation_factors2() -> None:
    accumulation_steps = (2, 4, 5)
    expected = [2, 2, 2, 2, 1]
    actual = list(training._accumulation_factors(accumulation_steps))

    assert actual == expected


def test_accumulation_factors4() -> None:
    accumulation_steps = (1,)
    expected = [1]
    actual = list(training._accumulation_factors(accumulation_steps))

    assert actual == expected


def test_accumulation_factors5() -> None:
    for i in range(1, 12):
        accumulation_steps = (i,)
        expected = [i] * i
        actual = list(training._accumulation_factors(accumulation_steps))

        assert actual == expected
