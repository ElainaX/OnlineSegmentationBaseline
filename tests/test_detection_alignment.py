import unittest
import numpy as np
from submap_disjoint.detection_alignment import refine_detection_pose


class TestDetectionAlignment(unittest.TestCase):
    def test_recovers_small_rigid_shift_and_improves_held_out(self):
        rng=np.random.default_rng(42)
        target=rng.uniform(-1,1,(5000,3));shift=np.array([.025,-.02,.01])
        source=target+shift
        pose,meta=refine_detection_pose(source,target,np.eye(4))
        self.assertTrue(meta['accepted'])
        np.testing.assert_allclose(pose[:3,3],-shift,atol=.005)

    def test_single_plane_is_not_enough_to_accept_pose_change(self):
        rng=np.random.default_rng(42);target=rng.uniform(-1,1,(5000,3));target[:,2]=1
        pose,meta=refine_detection_pose(target+np.array([.01,0,.01]),target,np.eye(4))
        self.assertFalse(meta['accepted'])
        np.testing.assert_equal(pose,np.eye(4))


if __name__=='__main__':unittest.main()
