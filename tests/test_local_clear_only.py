import unittest
import numpy as np
from submap_disjoint.local_clear_only import LocalClearCandidate
from submap_disjoint.readonly_detection_core import Candidate, CONTRADICTION, SUPPORT, UNKNOWN, SOURCE_DEPTH

CFG = dict(min_observation_gap=5, min_translation_m=.02, min_rotation_deg=3.,
    min_reference_tested_points=100, contradiction_ratio=.5, cancel_support_ratio=.6,
    cancel_observations=2, confirm_observations=2,
    min_cumulative_tested_fraction=.12, min_contradicted_fraction=.2)


class TestLocalClearOnly(unittest.TestCase):
    def test_only_spatial_clear_differs_counters_stay_identical(self):
        candidates = [cls(0, frozenset([0]), 0, 0, np.zeros((400,3)), np.zeros((400,3)))
                      for cls in (Candidate, LocalClearCandidate)]
        for frame, code in [(0, [CONTRADICTION]*100+[UNKNOWN]*300),
                             (5, [UNKNOWN]*100+[SUPPORT]*100+[UNKNOWN]*200)]:
            code=np.array(code);src=np.where(code==CONTRADICTION,SOURCE_DEPTH,0)
            pose=np.eye(4);pose[0,3]=frame*.01
            for c in candidates:c.observe(frame,pose,code,src,CFG)
        old,new=candidates
        self.assertEqual(old.hits,new.hits)
        self.assertEqual(old.support_hits,new.support_hits)
        self.assertEqual(old.state,new.state)
        self.assertEqual(old.contradicted.sum(),0)
        self.assertEqual(new.contradicted.sum(),100)

    def test_same_region_support_clears_both(self):
        for cls in (Candidate, LocalClearCandidate):
            c=cls(0,frozenset([0]),0,0,np.zeros((100,3)),np.zeros((100,3)))
            for frame,code in [(0,CONTRADICTION),(5,SUPPORT)]:
                codes=np.full(100,code);pose=np.eye(4);pose[0,3]=frame*.01
                c.observe(frame,pose,codes,np.where(codes==CONTRADICTION,SOURCE_DEPTH,0),CFG)
            self.assertEqual(c.contradicted.sum(),0)


if __name__=='__main__':unittest.main()
