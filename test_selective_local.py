import unittest
import numpy as np
from selective_local import choose_resolution, split_views, score_surface, assemble_outputs, observation_votes, refine_mask

class SelectiveTests(unittest.TestCase):
    def test_no_gain_keeps_ig(self):
        self.assertEqual(choose_resolution([.8]*3,[.81]*3,[.81]*3),'ig')
    def test_fine_must_beat_local_coarse(self):
        self.assertEqual(choose_resolution([.5]*3,[.8]*3,[.81]*3),'local25')
        self.assertEqual(choose_resolution([.5]*3,[.8]*3,[.85]*3),'local20')
    def test_single_view_gain_rejected(self):
        self.assertEqual(choose_resolution([.5]*3,[.9,.49,.49]),'ig')
    def test_disjoint_validation(self):
        views=[({'frame':i},None) for i in range(10)]
        train,val=split_views(views)
        self.assertEqual(len(val),3)
        self.assertFalse({p['frame'] for p,_ in train}&{p['frame'] for p,_ in val})
    def test_geometry_ownership(self):
        xyz=np.eye(3);rgb=np.zeros((3,3));m=np.array([[1,0],[1,1],[0,1]],bool)
        x,c,n=assemble_outputs(xyz,rgb,m,{0:(np.array([[4.,0,0]]),np.zeros((1,3)))})
        np.testing.assert_array_equal(x[:3],xyz)
        np.testing.assert_array_equal(n[:3,1],m[:,1])
        self.assertFalse(n[:3,0].any());self.assertTrue(n[3,0]);self.assertFalse(n[3,1])
    def test_depth_and_occlusion(self):
        K=np.array([[100.,0,2],[0,100.,2],[0,0,1]])
        p={'frame':0,'depth':np.ones((5,5)), 'pose':np.eye(4)}
        mask=np.ones((5,5),bool)
        good=np.array([[0.,0,1.]])
        self.assertGreater(score_surface(good,[(p,mask)],K)[0],.5)
        self.assertEqual(score_surface(np.array([[0.,0,.5]]),[(p,mask)],K)[0],0.)
        self.assertEqual(score_surface(np.array([[0.,0,2.]]),[(p,mask)],K)[0],0.)
    def test_unknown_and_occluded_do_not_remove(self):
        K=np.eye(3);K[0,0]=K[1,1]=100
        views=[({'frame':i,'depth':np.ones((2,2)),'pose':np.eye(4)},np.ones((2,2),bool)) for i in range(3)]
        pos,neg=observation_votes(np.array([[0.,0,2.],[0.,0,1.],[0.,0,.5]]),views,K)
        np.testing.assert_array_equal(pos,[0,3,0]);np.testing.assert_array_equal(neg,[0,0,3])
    def test_contradiction_and_label_ownership(self):
        m=np.array([1,1,0,0],bool);other=np.array([0,0,1,0],bool)
        result,a=refine_mask(m,other,np.ones(4,bool),np.array([0,1,3,3]),np.array([3,3,0,0]))
        np.testing.assert_array_equal(result,[0,1,0,1])
        self.assertEqual(a,dict(removed_points=1,added_points=1))

if __name__=='__main__':unittest.main()
