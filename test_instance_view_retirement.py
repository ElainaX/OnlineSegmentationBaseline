import sys,tempfile,unittest
import numpy as np
from types import SimpleNamespace as NS
sys.path.append('third_party/FCGF')
from submap_disjoint.instance_view_retirement import update_view_votes,InstanceViewIGRetirement
from submap_disjoint.instance_retirement import RetirementOptions
from test_direct_retirement import bank,reference
from test_instance_retirement import fixture


class ViewTests(unittest.TestCase):
    def setUp(self):
        self.state={};self.points=np.tile([0.,0.,1.],(100,1));self.keys=np.arange(100)
        self.cfg=dict(confirm_observations=2,min_observation_gap=5,min_translation_m=.02,min_rotation_deg=3.)
        self.opt=RetirementOptions()

    def step(self,frame,x=0.,free=70,support=0,source=(0,)):
        codes=np.zeros(100,np.uint8);codes[:free]=2;codes[free:free+support]=1
        pose=np.eye(4);pose[0,3]=x
        return update_view_votes(self.state,self.points,self.keys,source,codes,frame,pose,self.opt,self.cfg)

    def test_two_independent_views_required(self):
        self.assertFalse(self.step(0)[0]);self.assertTrue(self.step(5,.1)[0])

    def test_repeated_same_camera_is_not_new_evidence(self):
        self.step(0)
        self.assertFalse(self.step(5)[0]);self.assertFalse(self.step(10)[0])

    def test_partial_visibility_never_counts_as_whole_absence(self):
        self.step(0,free=20)
        self.assertFalse(self.step(5,.1,free=20)[0])

    def test_positive_support_revokes_pending_views(self):
        self.step(0)
        self.assertFalse(self.step(5,.1,free=70,support=10)[0])
        self.assertFalse(self.step(10,.2)[0]);self.assertTrue(self.step(15,.3)[0])

    def test_unknown_does_not_confirm_or_cancel(self):
        self.step(0);self.assertFalse(self.step(5,.1,free=0)[0])
        self.assertTrue(self.step(10,.2)[0])

    def test_reference_extension_needs_reconfirmation(self):
        self.step(0);self.keys=self.keys+1
        self.assertFalse(self.step(5,.1)[0]);self.assertTrue(self.step(10,.2)[0])

    def test_identity_extension_needs_reconfirmation(self):
        self.step(0)
        self.assertFalse(self.step(5,.1,source=(0,1))[0])

    def test_real_instance_removed_without_per_point_confirmation(self):
        o,ids,p=fixture();b=bank();b.streak[:]=1;b.last_conflict_frame[:]=5;b.last_frame=5
        cfg=dict(self.cfg,edge_threshold_m=.08,boundary_margin_px=0,depth_epsilon_m=.08,
                 depth_relative_epsilon=.01,min_map_weight=3,min_instance_weight=5)
        d=NS(bank=b,K=np.array([[100.,0,50],[0,100,50],[0,0,1]]),cfg=cfg,cam=dict(depth_near=.1,depth_far=5.))
        opt=RetirementOptions(min_reference_points=3,max_reference_points=3,min_confirmed_points=2,max_shared_fraction=.5,verify_state=True)
        with tempfile.TemporaryDirectory() as tmp:
            ctrl=InstanceViewIGRetirement(o,d,tmp,opt,True)
            ctrl.memberships=lambda p:[frozenset({0}) for _ in p]
            ctrl.reference=lambda row,source:reference(b,[0,1,2])
            self.assertEqual(ctrl.step(5,np.full((101,101),2.),np.eye(4)),[])
            b.last_frame=10;pose=np.eye(4);pose[0,3]=.1
            events=ctrl.step(10,np.full((101,101),2.),pose)
            self.assertEqual(len(events),1);self.assertEqual(o.global2merged_mask_id,[-1,0])
            self.assertEqual(events[0]['evidence']['original_point_confirmed_fraction'],0.)


if __name__=='__main__':unittest.main()
