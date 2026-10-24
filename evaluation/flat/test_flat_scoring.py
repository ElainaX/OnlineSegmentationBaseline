"""Analytic edge cases independent of either model output."""
import json
import numpy as np
from flat_scoring import correspondences, evaluate_projected, score_reconstruction
from protocol_utils import ROOT, sha


def main():
    passed = []
    labels = np.array([1] * 100 + [2] * 100 + [0] * 100)
    perfect = np.stack([labels == 1, labels == 2], axis=1)
    def score(masks, scores, minimum=100):
        return evaluate_projected(labels, [1, 2], masks, scores, minimum)
    assert score(perfect, [1., .8])["AP"] == 1.
    passed.append("perfect predictions AP=1")
    assert score(np.zeros((300, 0), bool), [])["AP"] == 0.
    passed.append("empty predictions with eligible GT AP=0")
    assert evaluate_projected(np.zeros(100, int), [], np.ones((100, 1), bool), [1.])["AP"] is None
    passed.append("no eligible GT is undefined")
    duplicate = np.column_stack([perfect, perfect[:, 0]])
    assert 0 < score(duplicate, [1., .8, .9])["AP"] < 1
    passed.append("higher-ranked duplicate penalized")
    stuff = np.column_stack([labels == 0, perfect])
    assert score(stuff, [1., .9, .8])["AP"] < 1.
    passed.append("known-stuff prediction is false positive")
    spill = perfect.copy()
    spill[labels == 0, 0] = True
    assert score(spill, [1., .8])["AP50"] < 1.
    passed.append("IoU exactly 0.5 fails strict threshold")
    small = np.array([3] * 9 + [0] * 10)
    assert evaluate_projected(small, [3], (small == 3)[:, None], [1.], 100)["AP"] is None
    assert evaluate_projected(small, [3], (small == 3)[:, None], [1.], 1)["AP"] == 1.
    passed.append("predeclared size strata differ without filling undefined")
    xyz = np.array([[0., 0., 0.], [2., 0., 0.]])
    assert correspondences(xyz, np.array([[0., 0., 0.]])).tolist() == [0, -1]
    assert correspondences(xyz, np.empty((0, 3))).tolist() == [-1, -1]
    assert correspondences(np.zeros((1, 3)), np.array([[.01, 0, 0], [-.01, 0, 0]])).tolist() == [0]
    passed.append("missing, empty and tied projection deterministic")
    domain = dict(xyz=xyz, instance_id=np.array([1, 2]), is_thing=np.ones(2, bool))
    result = score_reconstruction(domain, xyz[:1], np.ones((1, 1), bool), [1.])
    assert result["secondary_all_sizes"]["eligible_GT_ids"] == [1, 2]
    assert result["GT_points_without_reconstruction"] == 1
    assert result["secondary_all_sizes"]["AP"] < 1.
    passed.append("unreconstructed GT retained in recall denominator")
    # Groundtruth perfect masks are test fixtures only; no mapping inputs are changed.
    report = dict(status="CPU_scoring_analytic_tests_passed", tests=passed,
        model_predictions_used=False, protocol_sha256=sha(ROOT / "scoring_protocol.json"))
    (ROOT / "scoring_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
