from ...attacking import pipeline


def test_calc_worst_score_initial():
    actual = pipeline.calc_worst_score(0.5, 0.3, 0.7, 0.8)
    assert actual == 0.8


def test_calc_worst_score_inside_bounds():
    actual = pipeline.calc_worst_score(0.5, 0.4, 0.6, 0.55)
    assert actual == 0.55


def test_calc_worst_score_better_score():
    actual = pipeline.calc_worst_score(0.55, 0.4, 0.6, 0.5)
    assert actual == 0.55
