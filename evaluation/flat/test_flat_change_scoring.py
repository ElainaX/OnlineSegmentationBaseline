import json
import tempfile
from pathlib import Path
import numpy as np
from flat_change_scoring import evaluate_actions, set_score, supported_surface
from protocol_utils import ROOT, sha


def main():
    checks = []
    t = np.array([True, False, False])
    n = np.array([False, True, False])
    no = np.zeros(3, bool)
    assert set_score(t, n, no, no)["D_observed"] == 0.
    assert set_score(t, n, no, t)["D_observed"] == 1.
    assert set_score(t, n, t, no)["D_observed"] == 0.
    assert set_score(t, n, t, t)["D_observed"] == 1.
    assert set_score(t, n, no, t | n)["D_observed"] == .5
    assert set_score(no, no, no, no)["D_observed"] is None
    checks.append("no actions, perfect, premature, duplicate, known-negative and undefined arithmetic")
    with tempfile.TemporaryDirectory(prefix="flat_evidence_test_") as temp:
        root = Path(temp)
        (root / "correction").mkdir()
        a = root / "correction/400.npz"
        e = a.with_suffix(".evidence.npz")
        np.savez(a, xyz=np.array([[0., 0., 0.]]), indices=np.array([1]), tsdf=np.array([-.5]), weight=np.array([1.]))
        np.savez(e, xyz=np.array([[0., 0., 0.], [.025, 0., 0.]]), indices=np.array([1, 2]),
            tsdf=np.array([-.5, .5]), weight=np.ones(2), erased=np.array([True, False]))
        surf, audit = supported_surface(a, e)
        assert np.allclose(surf, [[.0125, 0., 0.]])
        checks.append("crossing incident on erased node includes surviving neighbour")
        domain = dict(xyz=np.array([[.0125, 0., 0.], [1., 0., 0.], [2., 0., 0.]]),
            target=t, negative=n, neutral=~(t | n), instance_id=np.array([6, 0, -1]))
        empty = np.empty((0, 3))
        assert evaluate_actions(root, domain, empty, "baseline")["D_observed"] == 0.
        assert evaluate_actions(root, domain, empty, "H")["status"] == "unavailable"
        checks.append("baseline no correction distinct from missing H evidence")
        summary = root / "correction/summary.json"
        action = dict(frame=400, archive=str(a), voxels=1)
        summary.write_text(json.dumps({"actions": [action, action]}))
        assert evaluate_actions(root, domain, empty, "H")["TP"] == 1
        assert evaluate_actions(root, domain, domain["xyz"][:1], "H")["TP"] == 0
        checks.append("union across actions; reconstructed-again surface earns no credit")
        summary.write_text(json.dumps({"actions": [action | {"frame": 0}]}))
        result = evaluate_actions(root, domain, empty, "H")
        assert result["TP"] == 0 and result["FP"] == 1 and result["FN"] == 1
        checks.append("early-only physical erasure is not successful adaptation")
        summary.write_text(json.dumps({"actions": [action | {"archive": str(root / "missing.npz")}]}))
        assert evaluate_actions(root, domain, empty, "H")["D_observed"] is None
        checks.append("missing archive not treated as no action")
        np.savez(e, xyz=np.array([[0., 0., 0.]]), indices=np.array([1]), tsdf=np.array([.5]),
            weight=np.ones(1), erased=np.ones(1, bool))
        summary.write_text(json.dumps({"actions": [action]}))
        assert evaluate_actions(root, domain, empty, "H")["D_observed"] is None
        checks.append("mismatched pre-erasure evidence rejected")
    report = dict(status="passed", tests=checks, actual_model_evidence_validated=False,
                  protocol_sha256=sha(ROOT / "change_scoring_protocol.json"))
    (ROOT / "change_scoring_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
