import unittest
import numpy as np
from submap_disjoint.ig_visible_evidence import VisibleOptions,TwoViewEvidence,project_evidence
from submap_disjoint.contradiction import UNKNOWN,SUPPORT,FREE,OCCLUDED

class VisibleTests(unittest.TestCase):
    def setUp(self):
        self.opt=VisibleOptions();self.pose=np.eye(4);self.K=np.array([[100.,0,20],[0,100,20],[0,0,1.]])
        u,v=np.meshgrid(np.arange(5,35,3),np.arange(5,35,3))
        self.points=np.c_[(u.ravel()-20)/100,(v.ravel()-20)/100,np.ones(u.size)]

    def test_current_denominator_ignores_unseen_reference(self):
        depth=np.full((40,40),2.)
        hidden=np.c_[np.full(1000,100.),np.zeros(1000),np.ones(1000)]
        _,a=project_evidence(self.points,depth,self.K,self.pose,.1,5,self.opt)
        _,b=project_evidence(np.r_[self.points,hidden],depth,self.K,self.pose,.1,5,self.opt)
        self.assertEqual(a['visible_conflict_ratio'],1.)
        self.assertEqual(a['visible_cells'],b['visible_cells']);self.assertTrue(b['candidate'])

    def test_occlusion_and_holes_are_not_free(self):
        for d,expected in [(.5,OCCLUDED),(0,UNKNOWN),(1,SUPPORT),(2,FREE)]:
            codes,row=project_evidence(self.points,np.full((40,40),d),self.K,self.pose,.1,5,self.opt)
            self.assertTrue(np.all(codes==expected))
            if expected in (UNKNOWN,OCCLUDED):self.assertEqual(row['visible_cells'],0);self.assertFalse(row['candidate'])

    def test_projection_uses_nearest_layer_per_pixel(self):
        p=np.r_[self.points,self.points*2]
        codes,r=project_evidence(p,np.full((40,40),3.),self.K,self.pose,.1,5,self.opt)
        self.assertEqual(r['visible_cells'],len(self.points));self.assertTrue(np.all(codes[len(self.points):]==UNKNOWN))

    def test_same_pose_never_confirms(self):
        mem=TwoViewEvidence(40,self.opt);codes=np.full(40,FREE,np.uint8)
        for i in range(100):self.assertFalse(mem.update(i,self.pose,codes).any())

    def test_same_cells_two_views_and_support_retracts(self):
        mem=TwoViewEvidence(40,self.opt);codes=np.full(40,FREE,np.uint8);p=self.pose.copy();p[0,3]=.2
        self.assertFalse(mem.update(0,self.pose,codes).any());self.assertTrue(mem.update(1,p,codes).all())
        self.assertFalse(mem.update(2,p,np.full(40,SUPPORT,np.uint8)).any())
        self.assertFalse(mem.update(3,self.pose,codes).any())

    def test_disjoint_patches_do_not_make_two_votes(self):
        mem=TwoViewEvidence(80,self.opt);p=self.pose.copy();p[0,3]=.2
        a=np.r_[np.full(40,FREE),np.zeros(40)].astype(np.uint8);b=a[::-1].copy()
        self.assertFalse(mem.update(0,self.pose,a).any());self.assertFalse(mem.update(1,p,b).any())

    def test_history_alone_never_authorizes_current_erasure(self):
        mem=TwoViewEvidence(40,self.opt);p=self.pose.copy();p[0,3]=.2;c=np.full(40,FREE,np.uint8)
        mem.update(0,self.pose,c);mem.update(1,p,c)
        self.assertFalse(mem.update(2,p,np.full(40,OCCLUDED,np.uint8)).any())
        self.assertFalse(mem.update(3,p,np.full(40,UNKNOWN,np.uint8)).any())

    def test_expired_free_needs_fresh_confirmation(self):
        mem=TwoViewEvidence(40,self.opt);p=self.pose.copy();p[0,3]=.2;c=np.full(40,FREE,np.uint8)
        mem.update(0,self.pose,c)
        self.assertFalse(mem.update(64,p,c).any())

if __name__=='__main__':unittest.main()
