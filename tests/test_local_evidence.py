import unittest
import numpy as np
from submap_disjoint.local_evidence import EvidenceBank, project_codes, connected_regions
from submap_disjoint.readonly_detection_core import SUPPORT, CONTRADICTION, UNKNOWN, OCCLUDED

CFG = dict(min_observation_gap=5, min_translation_m=.02, min_rotation_deg=3.,
           depth_epsilon_m=.08, depth_relative_epsilon=.01)


def pose(x=0):
    p = np.eye(4)
    p[0, 3] = x
    return p


class TestLocalEvidence(unittest.TestCase):
    def bank(self):
        b = EvidenceBank(.025, CFG, 10)
        b.seed([[0, 0, 2], [.1, 0, 2]], 0)
        return b

    def test_other_surface_support_does_not_erase_conflict(self):
        b = self.bank()
        b.observe(0, pose(), [CONTRADICTION, SUPPORT])
        b.observe(5, pose(.03), [CONTRADICTION, SUPPORT])
        self.assertEqual(b.confirmed[:2].tolist(), [True, False])
        self.assertEqual(b.score[:2].tolist(), [2, -2])

    def test_no_repeat_votes_same_view_or_same_frame(self):
        b = self.bank()
        b.observe(0, pose(), [CONTRADICTION] * 2)
        b.observe(0, pose(.1), [CONTRADICTION] * 2)
        b.observe(8, pose(), [CONTRADICTION] * 2)
        self.assertEqual(b.contradictions[:2].tolist(), [1, 1])
        self.assertFalse(b.confirmed[:2].any())

    def test_occluded_and_unknown_do_not_vote(self):
        b = self.bank()
        b.observe(0, pose(), [CONTRADICTION] * 2)
        b.observe(5, pose(.03), [OCCLUDED, UNKNOWN])
        self.assertEqual(b.score[:2].tolist(), [1, 1])
        b.observe(10, pose(.06), [CONTRADICTION] * 2)
        self.assertTrue(b.confirmed[:2].all())

    def test_confirmation_can_be_revoked_and_reconfirmed(self):
        b = self.bank()
        for f, x, code in [(0, 0, CONTRADICTION), (5, .03, CONTRADICTION),
                            (10, .06, SUPPORT), (15, .09, SUPPORT)]:
            b.observe(f, pose(x), [code] * 2)
        self.assertFalse(b.confirmed[:2].any())
        self.assertEqual(b.transitions, dict(confirmed=2, revoked=2))
        b.observe(20, pose(.12), [CONTRADICTION] * 2)
        b.observe(25, pose(.15), [CONTRADICTION] * 2)
        self.assertTrue(b.confirmed[:2].all())

    def test_cell_dedup_no_vote_inflation(self):
        b = self.bank()
        b.seed([[.001, .001, 2.001]] * 100, 1)
        self.assertEqual(b.n, 2)

    def test_free_space_tests_multiple_old_layers(self):
        K = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1.]])
        p = np.array([[0, 0, 1.], [0, 0, 2.], [0, 0, 4.]])
        depth = np.full((3, 3), 3.)
        codes = project_codes(p, K, pose(), depth, np.ones((3, 3), bool), CFG, .1, 8.)
        self.assertEqual(codes.tolist(), [CONTRADICTION, CONTRADICTION, OCCLUDED])
        codes = project_codes(p, K, pose(), depth, np.zeros((3, 3), bool), CFG, .1, 8.)
        self.assertEqual(codes.tolist(), [UNKNOWN] * 3)

    def test_connected_region_is_not_instance_merge(self):
        keys = np.array([[0, 0, 0], [1, 1, 1], [8, 0, 0]])
        regions = connected_regions(keys, np.array([0, 1, 2]))
        self.assertEqual(sorted(map(len, regions)), [1, 2])

    def test_capacity_reports_dropped_cells(self):
        b = EvidenceBank(.025, CFG, 1)
        b.seed([[0, 0, 2], [1, 0, 2]], 0)
        self.assertEqual(b.n, 1)
        self.assertEqual(b.dropped, 1)


if __name__ == '__main__':
    unittest.main()
