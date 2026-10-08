import unittest
import numpy as np
from submap_disjoint.surface_bank import SurfaceBank, voxel_keys

CFG=dict(depth_epsilon_m=.08,depth_relative_epsilon=.01,
    min_observation_gap=5,min_translation_m=0.,min_rotation_deg=0.,confirm_observations=2)


class TestSurfaceBank(unittest.TestCase):
    def test_exist_then_disappear_then_return(self):
        bank=SurfaceBank(.025,CFG,seed_stride=1)
        k=np.array([[100,0,2],[0,100,2],[0,0,1]],float)
        depth=np.ones((5,5));interior=np.ones((5,5),bool)
        bank.step(0,np.eye(4),k,depth,interior,.1,5)
        n=len(bank.points)
        bank.step(5,np.eye(4),k,depth,interior,.1,5)
        bank.step(10,np.eye(4),k,depth*2,interior,.1,5)
        self.assertFalse(bank.active[:n].any())
        bank.step(15,np.eye(4),k,depth*2,interior,.1,5)
        self.assertTrue(bank.active[:n].all())
        bank.step(20,np.eye(4),k,depth,interior,.1,5)
        self.assertFalse(bank.active[:n].any())

    def test_single_old_measurement_and_unknown_do_not_confirm(self):
        bank=SurfaceBank(.025,CFG,seed_stride=1);k=np.eye(3)
        depth=np.ones((2,2));valid=np.ones((2,2),bool)
        bank.step(0,np.eye(4),k,depth,valid,.1,5)
        bank.step(5,np.eye(4),k,depth*2,valid,.1,5)
        bank.step(10,np.eye(4),k,depth*2,valid,.1,5)
        bank.step(15,np.eye(4),k,depth*0,valid&False,.1,5)
        self.assertFalse(bank.active.any())

    def test_voxel_keys_do_not_alias_negative_coordinates(self):
        points=np.array([[-.01,0,0],[0,0,0],[.03,0,0],[0,-.01,0]])
        self.assertEqual(len(np.unique(voxel_keys(points,.025))),4)

    def test_color_change_alone_cannot_remove_but_can_corroborate_depth(self):
        cfg=dict(CFG,rgb_mad_multiplier=4,rgb_threshold_min=8,rgb_threshold_max=50)
        bank=SurfaceBank(.025,cfg,seed_stride=1,require_color=True)
        k=np.array([[100,0,2],[0,100,2],[0,0,1]],float)
        depth=np.ones((5,5));valid=np.ones((5,5),bool)
        red=np.zeros((5,5,3),np.uint8);red[:,:,0]=255
        blue=red[:,:,::-1].copy()
        bank.step(0,np.eye(4),k,depth,valid,.1,5,red);n=len(bank.points)
        bank.step(5,np.eye(4),k,depth,valid,.1,5,red)
        bank.step(10,np.eye(4),k,depth,valid,.1,5,blue)
        self.assertFalse(bank.active.any())
        bank.step(15,np.eye(4),k,depth*2,valid,.1,5,blue)
        bank.step(20,np.eye(4),k,depth*2,valid,.1,5,blue)
        self.assertTrue(bank.active[:n].all())


if __name__=='__main__':unittest.main()
