import sys
import tempfile
import unittest
from types import SimpleNamespace as NS
import numpy as np
sys.path.append('third_party/FCGF')
from submap_disjoint.persistent_retirement import (surface_reference,init_state,union_states,
    update_state,resolve_live_track,PersistentIGInstanceRetirement)
from submap_disjoint.instance_retirement import RetirementOptions,DepthContext
from test_instance_retirement import fixture


def bank(points,streak=2):
    n=len(points)
    return NS(points=np.asarray(points,float),voxel_size=.025,support=np.full(n,3),
              streak=np.full(n,streak),last_conflict_frame=np.full(n,10))


class PersistentEvidenceTests(unittest.TestCase):
    def state(self,streak=2,inherit=True):
        b=bank([[0,0,1]],streak)
        r=surface_reference(b.points,b,.04375,2)
        return b,init_state(r,b,10,inherit)

    def test_duplicate_corners_are_one_anchor_vote(self):
        b=bank([[0,0,1]])
        r=surface_reference([[0,0,1],[.02,0,1],[0,.02,1]],b,.04375,2)
        self.assertEqual(len(r['points']),1)
        np.testing.assert_array_equal(r['points'],b.points)

    def test_unknown_reference_area_not_discarded(self):
        b=bank([[0,0,1]])
        r=surface_reference([[0,0,1],[1,0,1]],b,.04375,2)
        self.assertEqual(len(r['points']),2);self.assertEqual(sum(r['anchors']<0),1)

    def test_opposing_corner_vote_removed_by_same_surface_location(self):
        b=bank([[0,0,1.88]])
        corner=np.array([[0,0,1.9125]])
        r=surface_reference(corner,b,.04375,2)
        cfg=dict(edge_threshold_m=.08,boundary_margin_px=0,depth_epsilon_m=.08,depth_relative_epsilon=.01)
        ctx=DepthContext(np.full((11,11),2.),np.eye(4),np.array([[100,0,5],[0,100,5],[0,0,1.]]),cfg,.1,5,.05)
        self.assertEqual(ctx.classify(corner)[0],1)
        self.assertEqual(ctx.classify(r['points'])[0],2)

    def test_unknown_occlusion_and_long_timeout_keep_confirmation(self):
        b,c=self.state()
        for frame,code in [(70,0),(200,3),(500,0)]:
            self.assertTrue(update_state(c,b,np.array([code]),frame,2)[0])

    def test_support_revokes_and_requires_two_new_detector_votes(self):
        b,c=self.state()
        self.assertFalse(update_state(c,b,np.array([1]),11,2)[0])
        b.streak[:]=3
        self.assertFalse(update_state(c,b,np.array([2]),20,2)[0])
        b.streak[:]=4
        self.assertTrue(update_state(c,b,np.array([2]),30,2)[0])

    def test_bank_support_reset_also_revokes(self):
        b,c=self.state();b.streak[:]=0
        self.assertFalse(update_state(c,b,np.array([0]),20,2)[0])

    def test_newly_acquired_area_cannot_inherit_old_active_bank(self):
        b,c=self.state(5,False)
        self.assertFalse(update_state(c,b,np.array([2]),11,2)[0])
        b.streak[:]=6;self.assertFalse(update_state(c,b,np.array([2]),20,2)[0])
        b.streak[:]=7;self.assertTrue(update_state(c,b,np.array([2]),30,2)[0])

    def test_repeated_same_frame_depth_cannot_manufacture_votes(self):
        b,c=self.state(1)
        for frame in [11,12,13,14]:
            self.assertFalse(update_state(c,b,np.array([2]),frame,2)[0])

    def test_union_preserves_old_and_keeps_added_area_unconfirmed(self):
        b,c=self.state();b2=bank([[1,0,1]])
        added=init_state(surface_reference(b2.points,b2,.04375,2),b2,20,False)
        u=union_states([c,added]);self.assertEqual(len(u['points']),2)
        self.assertEqual(sorted(u['local_streak']),[0,2])
        again=union_states([u,added]);self.assertEqual(len(again['points']),2)

    def test_union_newer_support_cannot_resurrect_old_confirmation(self):
        b,c=self.state();new=init_state(c,b,20,False);new['last_support'][:]=20
        u=union_states([c,new]);self.assertEqual(u['local_streak'][0],0)
        self.assertEqual(u['last_support'][0],20)

    def test_resolve_expansion_only_for_audit(self):
        self.assertEqual(resolve_live_track({0},[0,0]),(0,'stable'))
        self.assertEqual(resolve_live_track({0,1},[0,1])[1],'split_or_partially_removed')

    def test_controller_expansion_retains_candidate_and_audits_new_area(self):
        b,c=self.state();c.update(id=1,source=(0,),created=10,last_stats={})
        b.points=np.vstack([b.points,[1,0,1]]);b.support=np.array([3,3]);b.streak=np.array([2,5])
        detector=NS(bank=b)
        with tempfile.TemporaryDirectory() as tmp:
            controller=PersistentIGInstanceRetirement(NS(global2merged_mask_id=[0,0]),detector,tmp)
            controller.pending={(0,):c}
            controller.reference=lambda row,source:surface_reference(b.points,b,.04375,2)
            controller.synchronize(400)
            result=controller.pending[(0,1)]
            self.assertEqual(result['id'],1);self.assertEqual(result['created'],10)
            self.assertEqual(result['local_streak'].tolist(),[2,0])
            self.assertEqual(len(result['points']),2)

    def test_real_controller_retires_only_after_confirmation(self):
        o,ids,p=fixture();b=bank(p.numpy(),0)
        b.streak[:2]=1;b.last_conflict_frame[:]=5;b.last_frame=5
        cfg=dict(edge_threshold_m=.08,boundary_margin_px=0,depth_epsilon_m=.08,depth_relative_epsilon=.01,
                 confirm_observations=2,min_map_weight=3,min_instance_weight=5)
        detector=NS(bank=b,K=np.array([[100,0,5],[0,100,5],[0,0,1.]]),cfg=cfg,cam=dict(depth_near=.1,depth_far=5.))
        options=RetirementOptions(min_reference_points=3,max_reference_points=3,min_confirmed_points=2,max_shared_fraction=.5,verify_state=True)
        with tempfile.TemporaryDirectory() as tmp:
            controller=PersistentIGInstanceRetirement(o,detector,tmp,options,True)
            self.assertEqual(controller.step(5,np.full((11,11),2.),np.eye(4)),[])
            b.streak[:2]=2;b.last_conflict_frame[:2]=10;b.last_frame=10
            events=controller.step(10,np.full((11,11),2.),np.eye(4))
            self.assertEqual(len(events),1);self.assertEqual(o.global2merged_mask_id,[-1,0])


if __name__=='__main__':unittest.main()
