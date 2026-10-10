import numpy as np
from eval_gap_v2 import combine, score_deletions


def test_no_deletion_is_exactly_standard_ap():
    for ap in (0.0, 0.2, 1.0):
        out = combine(ap, 0.0)
        assert out['G_AP'] == ap
        assert out['bonus'] == 0.0


def test_deletion_bonus_is_nonnegative_and_bounded():
    out = combine(0.25, 0.5)
    assert out['G_AP'] == 0.625
    assert out['G_AP'] >= out['AP']
    assert out['G_AP'] <= 1.0


def test_unmatched_deletion_support_dilutes_quality_without_negative_score():
    matched = score_deletions(np.array([[1.0]]), np.array([10.0]), np.array([10.0]))
    noisy = score_deletions(np.array([[1.0], [0.0]]), np.array([10.0]), np.array([10.0, 10.0]))
    assert matched['score'] == 1.0
    assert noisy['score'] == 0.5
    assert noisy['score'] >= 0.0
