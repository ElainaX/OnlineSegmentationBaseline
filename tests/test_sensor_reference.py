import unittest
import numpy as np
from submap_disjoint.sensor_reference import sensor_projection, PointViewVotes

CFG=dict(depth_epsilon_m=.08,depth_relative_epsilon=.01,
         min_observation_gap=5,min_translation_m=.02,min_rotation_deg=3.)


class TestSensorReference(unittest.TestCase):
    def test_depth_sign_unknown_occlusion_and_zbuffer(self):
        points=np.array([[0,0,1],[0,0,2],[1,0,1],[-1,0,1],[0,1,1]],float)
        k=np.array([[1,0,2],[0,1,2],[0,0,1]],float)
        depth=np.ones((5,5));depth[2,2]=3;depth[2,3]=.5;depth[2,1]=0
        codes,_=sensor_projection(points,k,np.eye(4),depth,depth>0,CFG,.1,5)
        np.testing.assert_array_equal(codes,[2,0,3,0,1])

    def test_view_clock_is_per_point(self):
        p=PointViewVotes(2)
        pose=np.eye(4);p.observe(0,pose,np.array([1,0]),CFG)
        pose=pose.copy();pose[0,3]=.03;p.observe(4,pose,np.array([0,1]),CFG)
        pose=pose.copy();pose[0,3]=.06;p.observe(5,pose,np.array([1,1]),CFG)
        np.testing.assert_array_equal(p.support,[2,1])

    def test_same_pose_unknown_and_occlusion_do_not_add_votes(self):
        p=PointViewVotes(3);pose=np.eye(4)
        p.observe(0,pose,np.array([1,3,0]),CFG)
        p.observe(10,pose,np.array([1,3,0]),CFG)
        np.testing.assert_array_equal(p.support,[1,0,0])
        self.assertEqual(p.conflict.sum(),0)


if __name__=='__main__':unittest.main()
