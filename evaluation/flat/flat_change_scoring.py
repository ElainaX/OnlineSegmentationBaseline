"""Offline physical-action evaluation on a fixed, partially observed Flat domain."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from reference_erasure_surface import archived_surface

RADIUS = np.sqrt(3) * .025


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def distances(a, b):
    return cKDTree(b).query(a)[0] if len(b) else np.full(len(a), np.inf)


def set_score(target, negative, before, after):
    tp = int((target & after).sum())
    fp = int(((before | after) & negative).sum() + (before & target & ~after).sum())
    fn = int(target.sum()) - tp
    den = tp + fp + fn
    return dict(D_observed=tp / den if den else None, TP=tp, FP=fp, FN=fn,
        target_points=int(target.sum()), union_points=den)


def supported_surface(archive, evidence):
    with np.load(archive, allow_pickle=False) as actual, np.load(evidence, allow_pickle=False) as z:
        for name in ["xyz", "indices", "tsdf", "weight"]:
            assert name in actual and name in z, f"Missing archive field {name}"
        assert "erased" in z, "Missing actual-erasure membership"
        selected = np.flatnonzero(z["erased"])
        selected = selected[np.argsort(z["indices"][selected])]
        order = np.argsort(actual["indices"])
        assert len(np.unique(actual["indices"])) == len(order)
        for name in ["indices", "tsdf", "weight"]:
            a, b = actual[name][order], z[name][selected]
            if name in ["tsdf", "weight"]:
                a, b = a.reshape(-1), b.reshape(-1)
            assert np.array_equal(a, b), f"Evidence differs from actual archive: {name}"
        # Archive XYZ can be float32 whereas one-ring positions are float64.
        # Require the same lattice nodes, allowing only representation rounding.
        a, b = actual["xyz"][order], z["xyz"][selected]
        assert np.allclose(a, b, rtol=0, atol=1e-5), "Evidence node position mismatch"
        assert np.array_equal(np.rint(a / .025), np.rint(b / .025)), "Evidence lattice mismatch"
        assert np.isfinite(z["xyz"]).all() and np.isfinite(z["tsdf"]).all()
        surface, audit = archived_surface(z)
        return surface, audit


def evaluate_actions(root, domain, final_points, method):
    assert method in ["baseline", "H"]
    root = Path(root).resolve()
    final_points = np.asarray(final_points)
    assert final_points.ndim == 2 and final_points.shape[1] == 3 and np.isfinite(final_points).all()
    ref = domain["xyz"]
    target, neutral, negative = (np.asarray(domain[k], bool) for k in ["target", "neutral", "negative"])
    assert np.all(target.astype(int) + neutral + negative == 1)
    summary_path = root / "correction/summary.json"
    if not summary_path.exists() and method == "H":
        return dict(status="unavailable", D_observed=None, reason="H correction summary missing")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"actions": []}
    if method == "baseline" and summary["actions"]:
        return dict(status="unavailable", D_observed=None, reason="Unexpected correction actions in original baseline")
    before, after = np.zeros(len(ref), bool), np.zeros(len(ref), bool)
    absent = distances(ref, final_points) > .05
    extra, unknown = set(), set()
    audits, hashes = [], {}
    if summary_path.exists():
        hashes[str(summary_path)] = digest(summary_path)
    try:
        for action in summary["actions"]:
            if not action.get("archive"):
                assert action.get("voxels", 0) == 0, "Physical deletion without archive"
                continue
            archive = Path(action["archive"])
            if not archive.is_absolute():
                archive = root / archive
            archive = archive.resolve()
            assert archive.is_relative_to(root), "Archive is outside this method output root"
            evidence = archive.with_suffix(".evidence.npz")
            assert archive.is_file() and evidence.is_file(), "Missing physical-action evidence"
            surf, audit = supported_surface(archive, evidence)
            hashes[str(archive)], hashes[str(evidence)] = digest(archive), digest(evidence)
            frame = int(action["frame"])
            assert 0 <= frame < 762, "Action frame outside concatenated source stream"
            gone = surf[distances(surf, final_points) > .05]
            near = distances(ref, gone) <= RADIUS
            hit = near & absent & ~neutral
            if frame >= 400:
                after |= hit
            else:
                before |= hit
            unknown.update(np.flatnonzero(near & absent & neutral).tolist())
            far = gone[distances(gone, ref) > RADIUS]
            extra.update(map(tuple, np.floor(far / .025).astype(np.int64)))
            audit.update(frame=frame, archive=str(archive), absent_surface_atoms=len(gone),
                known_reference_points_affected=int(hit.sum()))
            audits.append(audit)
    except (AssertionError, OSError, KeyError, ValueError) as exc:
        return dict(status="unavailable", D_observed=None, reason=str(exc), actions=audits,
                    archive_sha256=hashes)
    result = set_score(target, negative, before, after)
    per_object = []
    for ident in np.unique(domain["instance_id"][target]):
        t = target & (domain["instance_id"] == ident)
        per_object.append(dict(instance_id=int(ident), target_points=int(t.sum()),
            correctly_cleared_points=int((t & after).sum())))
    return dict(status="available_on_observed_domain", **result, per_object=per_object,
        neutral_reference_points_affected=len(unknown), off_reference_atoms_unscored=len(extra),
        early_only_points=int((before & ~after).sum()), archive_sha256=hashes, actions=audits,
        official_metric=False)
