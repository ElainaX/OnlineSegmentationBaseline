import tempfile
import unittest
from pathlib import Path
import numpy as np
from submap_disjoint.fixed_candidate_memory import FixedCandidateObserver
from submap_disjoint.readonly_detection_core import Candidate, CONTRADICTION, SUPPORT, UNKNOWN, SOURCE_DEPTH

CFG = dict(min_observation_gap=5, min_translation_m=.02, min_rotation_deg=3.,
    min_reference_tested_points=100, contradiction_ratio=.5, cancel_support_ratio=.6,
    cancel_observations=2, confirm_observations=2,
    min_cumulative_tested_fraction=.12, min_contradicted_fraction=.2)


class TestFixedCandidates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.o = FixedCandidateObserver(Path(self.tmp.name), CFG)
        self.c = Candidate(0, frozenset([4]), 2, 0, np.zeros((200, 3)), np.zeros((200, 3)))

    def tearDown(self):
        self.tmp.cleanup()

    def step(self, frame, first, second, x=None):
        codes = np.array([first] * 100 + [second] * 100, np.uint8)
        sources = np.where(codes == CONTRADICTION, SOURCE_DEPTH, 0).astype(np.uint8)
        pose = np.eye(4); pose[0, 3] = frame * .01 if x is None else x
        self.o.observe(self.c, frame, pose, codes, sources)
        if self.c.state == 'suspected':
            self.c.observe(frame, pose, codes, sources, CFG)
            self.o.verify_original(self.c)

    def test_same_observations_local_memory_survives_other_part_support(self):
        self.step(0, CONTRADICTION, UNKNOWN)
        self.step(5, UNKNOWN, SUPPORT)
        self.step(10, CONTRADICTION, UNKNOWN)
        self.assertEqual(self.c.state, 'suspected')
        self.assertTrue(self.o.records[0]['common'].confirmed)
        self.assertEqual(self.o.records[0]['common'].accepted_frames, [0, 5, 10])

    def test_same_part_support_retracts_its_evidence(self):
        self.step(0, CONTRADICTION, UNKNOWN)
        self.step(5, SUPPORT, UNKNOWN)
        self.step(10, CONTRADICTION, UNKNOWN)
        self.assertFalse(self.o.records[0]['common'].confirmed)

    def test_repeated_pose_does_not_add_vote(self):
        self.step(0, CONTRADICTION, UNKNOWN, x=0)
        self.step(5, CONTRADICTION, UNKNOWN, x=0)
        self.assertEqual(self.o.records[0]['common'].accepted_frames, [0])

    def test_continued_observation_separate_from_common(self):
        self.step(0, CONTRADICTION, UNKNOWN)
        self.step(5, UNKNOWN, SUPPORT)
        self.step(10, UNKNOWN, SUPPORT)
        self.assertEqual(self.c.state, 'cancelled')
        self.step(15, CONTRADICTION, UNKNOWN)
        self.assertFalse(self.o.records[0]['common'].confirmed)
        self.assertTrue(self.o.records[0]['extended'].confirmed)
        self.assertEqual(self.o.records[0]['common'].accepted_frames, [0, 5, 10])

    def test_disjoint_once_seen_parts_do_not_prove_repeat_confirmation(self):
        self.step(0, CONTRADICTION, UNKNOWN)
        self.step(5, UNKNOWN, CONTRADICTION)
        self.assertEqual(self.c.state, 'would_remove')
        self.assertFalse(self.o.records[0]['common'].confirmed)

    def test_archive_replays_original_and_preserves_candidate_identity(self):
        self.step(0, CONTRADICTION, UNKNOWN)
        self.step(5, CONTRADICTION, UNKNOWN)
        s = self.o.finalize([self.c])
        self.assertEqual(s['candidates'], 1)
        a = np.load(Path(self.tmp.name) / 'fixed_candidate_memory/0000.npz')
        np.testing.assert_array_equal(a['reference_points'], self.c.points)
        self.assertTrue(s['observation_sets_identical'])


if __name__ == '__main__':
    unittest.main()
