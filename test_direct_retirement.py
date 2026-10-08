import sys
import tempfile
import unittest
from types import SimpleNamespace as NS
import numpy as np
sys.path.append('third_party/FCGF')
from submap_disjoint.direct_retirement import DirectIGInstanceRetirement,union_direct,lease_authorized
from submap_disjoint.persistent_retirement import init_state,update_state
from submap_disjoint.instance_retirement import RetirementOptions
from submap_disjoint.surface_bank import voxel_keys
from test_instance_retirement import fixture


def bank():
    p=np.array([[0.,0,1.],[.025,0,1.],[.05,0,1.]])
    return NS(points=p,keys=voxel_keys(p,.025),voxel_size=.025,support=np.full(3,3),
              streak=np.array([2,2,0]),last_conflict_frame=np.array([10,10,-1]),last_frame=10)


def reference(b,ids):
    ids=np.asarray(ids,int)
    return dict(points=b.points[ids],anchors=ids,keys=b.keys[ids])


class DirectTests(unittest.TestCase):
    def test_receipt_requires_complete_original_ownership(self):
        self.assertTrue(lease_authorized({1,2},{0,1,2}))
        self.assertFalse(lease_authorized({1,2},{1}))
        self.assertFalse(lease_authorized(set(),{1}))
        self.assertFalse(lease_authorized(None,{1}))

    def test_unknown_surface_upgrades_without_double_count(self):
        b=bank();r=reference(b,[0]);r['anchors'][:]=-1
        unknown=init_state(r,b,5,False);direct=init_state(reference(b,[0]),b,10,True)
        out=union_direct([unknown,direct])
        self.assertEqual(len(out['points']),1);self.assertEqual(out['anchors'][0],0)
        self.assertEqual(out['local_streak'][0],2)

    def test_newer_support_prevents_inherited_confirmation(self):
        b=bank();old=init_state(reference(b,[0]),b,5,True)
        new=init_state(reference(b,[0]),b,10,False);new['last_support'][:]=10
        out=union_direct([old,new])
        self.assertFalse(update_state(out,b,np.array([0]),11,2)[0])

    def test_confirmed_id_attaches_even_when_tsdf_cell_is_gone(self):
        b=bank();o=NS(global2merged_mask_id=[0]);d=NS(bank=b)
        with tempfile.TemporaryDirectory() as tmp:
            ctrl=DirectIGInstanceRetirement(o,d,tmp)
            ctrl.receipts={0:frozenset({0}),1:frozenset({9})}
            r=reference(b,[2]);c=init_state(r,b,5,False)
            c.update(id=1,source=(0,),created=5)
            ctrl.attach_receipts(c,10)
            self.assertIn(0,c['anchors']);self.assertNotIn(1,c['anchors'])
            pos=list(c['anchors']).index(0)
            self.assertEqual(c['local_streak'][pos],2)
            ctrl.attach_receipts(c,11);self.assertEqual(len(c['points']),2)

    def test_first_conflict_receipt_survives_local_membership_erasure(self):
        b=bank();b.streak[:2]=1;b.last_frame=5;b.last_conflict_frame[:2]=5
        with tempfile.TemporaryDirectory() as tmp:
            ctrl=DirectIGInstanceRetirement(NS(global2merged_mask_id=[0]),NS(bank=b),tmp)
            ctrl.memberships=lambda p:[frozenset({0}) for _ in p]
            ctrl.receive(5)
            ctrl.memberships=lambda p:[frozenset() for _ in p]
            b.streak[:2]=2;b.last_frame=10;b.last_conflict_frame[:2]=10
            rows=ctrl.receive(10)
            self.assertEqual(ctrl.receipts[0],frozenset({0}));self.assertEqual(rows[0],2)

    def test_support_revokes_receipt_and_new_conflict_recaptures_owner(self):
        b=bank()
        with tempfile.TemporaryDirectory() as tmp:
            ctrl=DirectIGInstanceRetirement(NS(global2merged_mask_id=[0,1]),NS(bank=b),tmp)
            ctrl.receipts={0:frozenset({0})};ctrl.receipt_created={0:5}
            b.streak[:]=0;ctrl.receive(11);self.assertNotIn(0,ctrl.receipts)
            b.streak[0]=1;b.last_conflict_frame[0]=20
            ctrl.memberships=lambda p:[frozenset({1}) for _ in p]
            ctrl.receive(20);self.assertEqual(ctrl.receipts[0],frozenset({1}))

    def test_no_inherited_votes_without_receipt(self):
        b=bank()
        with tempfile.TemporaryDirectory() as tmp:
            ctrl=DirectIGInstanceRetirement(NS(),NS(bank=b),tmp)
            c=ctrl.state(reference(b,[0]),{0},10)
            self.assertEqual(c['local_streak'][0],0)

    def test_real_mutation_after_registered_multiview_confirmation(self):
        o,ids,p=fixture();b=bank();b.streak[:2]=1;b.last_conflict_frame[:2]=5;b.last_frame=5
        cfg=dict(edge_threshold_m=.08,boundary_margin_px=0,depth_epsilon_m=.08,depth_relative_epsilon=.01,
                 confirm_observations=2,min_map_weight=3,min_instance_weight=5)
        d=NS(bank=b,K=np.array([[100,0,5],[0,100,5],[0,0,1.]]),cfg=cfg,cam=dict(depth_near=.1,depth_far=5.))
        opt=RetirementOptions(min_reference_points=3,max_reference_points=3,min_confirmed_points=2,max_shared_fraction=.5,verify_state=True)
        with tempfile.TemporaryDirectory() as tmp:
            ctrl=DirectIGInstanceRetirement(o,d,tmp,opt,True)
            ctrl.memberships=lambda p:[frozenset({0}) for _ in p]
            ctrl.reference=lambda row,source:reference(b,[0,1,2])
            self.assertEqual(ctrl.step(5,np.full((11,11),2.),np.eye(4)),[])
            b.streak[:2]=2;b.last_conflict_frame[:2]=10;b.last_frame=10
            events=ctrl.step(10,np.full((11,11),2.),np.eye(4))
            self.assertEqual(len(events),1);self.assertEqual(o.global2merged_mask_id,[-1,0])
            self.assertEqual(events[0]['erased_corners'],2)


if __name__=='__main__':unittest.main()
