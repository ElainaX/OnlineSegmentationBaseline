"""Behavioral checks for the faithful onlineSeg port, without reconstruction."""
import ast
import hashlib
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

from submap_disjoint import readonly_detection_core as d
from submap_disjoint.readonly_queries import ReadOnlyMembershipQueries

ROOT = Path(__file__).resolve().parents[1]


class ReadOnlyDetectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((ROOT/'config/ig_readonly_onlineSeg.yaml').read_text())['remove_detection']
        self.pose = np.eye(4)

    def candidate(self, n=1000):
        return d.Candidate(0, frozenset([0, 1]), 0, 0,
                           np.zeros((n, 3), np.float32), np.zeros((n, 3), np.float32))

    def observe(self, c, frame, codes):
        pose = self.pose.copy(); pose[0, 3] = frame * .01
        sources = np.where(codes == d.CONTRADICTION, d.SOURCE_DEPTH, d.SOURCE_NONE).astype(np.uint8)
        return c.observe(frame, pose, codes, sources, self.cfg)

    def test_source_and_query_bodies_are_unchanged(self):
        provenance = json.loads((ROOT/'config/ig_readonly_provenance.json').read_text())
        self.assertEqual(hashlib.sha256(Path(d.__file__).read_bytes()).hexdigest(), provenance['detector_sha256'])
        text = (ROOT/'submap_disjoint/readonly_queries.py').read_text()
        for name, sha in provenance['query_method_sha256'].items():
            node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == name)
            body = textwrap.dedent('\n'.join(text.splitlines()[node.lineno-1:node.end_lineno]))
            self.assertEqual(hashlib.sha256(body.encode()).hexdigest(), sha, name)
        d.checked_settings(self.cfg)

    def test_disjoint_visible_parts_accumulate_without_repeated_point_votes(self):
        c = self.candidate()
        a = np.zeros(1000, np.uint8); a[:120] = d.CONTRADICTION
        b = np.zeros(1000, np.uint8); b[120:240] = d.CONTRADICTION
        self.assertEqual(self.observe(c, 0, a)[0], 'contradiction')
        self.assertEqual(self.observe(c, 5, b)[0], 'would_remove')
        self.assertEqual(c.contradicted.sum(), 240)
        self.assertEqual(c.hits, 2)

    def test_support_resets_evidence_and_cancels_after_two_observations(self):
        c = self.candidate()
        contradiction = np.zeros(1000, np.uint8); contradiction[:120] = d.CONTRADICTION
        support = np.zeros(1000, np.uint8); support[:150] = d.SUPPORT
        self.observe(c, 0, contradiction)
        self.assertEqual(self.observe(c, 5, support)[0], 'support')
        self.assertEqual(c.contradicted.sum(), 0)
        self.assertEqual(c.tested.sum(), 0)
        self.assertEqual(self.observe(c, 10, support)[0], 'cancelled')

    def test_occlusion_and_unknown_do_not_expire_old_evidence(self):
        c = self.candidate()
        first = np.zeros(1000, np.uint8); first[:120] = d.CONTRADICTION
        self.observe(c, 0, first)
        self.observe(c, 100, np.full(1000, d.OCCLUDED, np.uint8))
        self.assertEqual(c.contradicted.sum(), 120)
        second = np.zeros(1000, np.uint8); second[120:240] = d.CONTRADICTION
        self.assertEqual(self.observe(c, 200, second)[0], 'would_remove')

    def test_independence_uses_frame_gap_and_full_rotation_including_roll(self):
        angle = np.radians(4)
        rolled = np.eye(4)
        rolled[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        self.assertFalse(d.independent_observation(rolled, self.pose, 4, 0, self.cfg))
        self.assertTrue(d.independent_observation(rolled, self.pose, 5, 0, self.cfg))
        self.assertFalse(d.independent_observation(self.pose, self.pose, 100, 0, self.cfg))

    def test_rgb_can_contradict_frozen_reference_without_live_surface(self):
        points = np.array([[0., 0., 1.]])
        red = np.array([[1., 0., 0.]])
        green = np.array([[[0., 1., 0.]]])
        history = dict(valid=np.zeros((1, 1), bool), depth=np.zeros((1, 1)))
        context = dict(threshold=8., current_lab=d.rgb_to_lab(green))
        codes, sources = d.project_reference(points, red, np.eye(3), self.pose,
            np.ones((1, 1)), history, np.full((1, 1), -1), np.ones((1, 1), bool),
            context, self.cfg, .1, 5.)
        self.assertEqual(int(codes[0]), d.CONTRADICTION)
        self.assertEqual(int(sources[0]), d.SOURCE_RGB)

    def test_live_membership_mapping_handles_merge_tombstone_and_ambiguity(self):
        table = [set([0, 1]), set([1, 2]), None]
        owner = SimpleNamespace(voxel_hash_table=SimpleNamespace(hash_table=table))
        query = ReadOnlyMembershipQueries(owner)
        before = [None if x is None else x.copy() for x in table]
        self.assertEqual(query.query_active_instance_memberships([0, 1, 2, -1], [3, 3, 5]),
                         [frozenset([3]), frozenset([3, 5]), frozenset(), frozenset()])
        self.assertEqual(query.query_active_instance_memberships([0, 1], [-1, 0, 0]),
                         [frozenset([0]), frozenset([0])])
        self.assertEqual(table, before)

    def test_raw_lineage_blocks_candidates_until_cooldown_expires(self):
        detector = object.__new__(d.RemoveDetector)
        detector.cfg = self.cfg
        c = self.candidate(); detector.candidates = [c]
        self.assertTrue(detector._blocking_candidates(frozenset([1, 5]), 0))
        c.state = 'cancelled'; c.cancelled_frame = 10
        self.assertTrue(detector._blocking_candidates(frozenset([1, 5]), 39))
        self.assertFalse(detector._blocking_candidates(frozenset([1, 5]), 40))
        self.assertFalse(detector._blocking_candidates(frozenset([8]), 11))


if __name__ == '__main__':
    unittest.main(verbosity=2)
