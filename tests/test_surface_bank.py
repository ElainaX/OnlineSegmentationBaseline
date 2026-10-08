import unittest
import numpy as np
from submap_disjoint.surface_bank import SurfaceBank, voxel_keys, independent_rays

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

    def test_rotation_in_place_is_not_new_ray_but_lateral_translation_is(self):
        cfg=dict(CFG,min_translation_m=.02,min_rotation_deg=3.)
        points=np.array([[0,0,2.]],float);origins=np.zeros((1,3));frames=np.array([0])
        self.assertFalse(independent_rays(points,origins,np.zeros(3),frames,10,cfg)[0])
        self.assertTrue(independent_rays(points,origins,np.array([.2,0,0]),frames,10,cfg)[0])
        self.assertFalse(independent_rays(points,origins,np.array([0,0,.2]),frames,10,cfg)[0])

    def test_nearby_measured_surface_vetoes_pixel_miss(self):
        k=np.array([[100,0,2],[0,100,2],[0,0,1]],float)
        depth=np.ones((5,5));valid=np.ones((5,5),bool)
        plain=SurfaceBank(.025,CFG,seed_stride=1)
        checked=SurfaceBank(.025,CFG,seed_stride=1,surface_support=True)
        for bank in [plain,checked]:
            bank.step(0,np.eye(4),k,depth,valid,.1,5)
            bank.step(5,np.eye(4),k,depth,valid,.1,5)
        missed=depth.copy();missed[2,2]=2
        for frame in [10,15]:
            for bank in [plain,checked]:bank.step(frame,np.eye(4),k,missed,valid,.1,5)
        self.assertTrue(plain.active.any())
        self.assertFalse(checked.active.any())


if __name__=='__main__':unittest.main()
