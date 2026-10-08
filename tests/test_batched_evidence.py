import unittest
import numpy as np
from submap_disjoint.batched_evidence import BatchedCandidate
from submap_disjoint.local_clear_only import LocalClearCandidate
from submap_disjoint.readonly_detection_core import SUPPORT, CONTRADICTION, UNKNOWN, OCCLUDED, SOURCE_DEPTH

CFG = dict(min_observation_gap=5, min_translation_m=.02, min_rotation_deg=3.,
    min_reference_tested_points=100, contradiction_ratio=.5, cancel_support_ratio=.6,
    cancel_observations=2, confirm_observations=2,
    min_cumulative_tested_fraction=.12, min_contradicted_fraction=.2)


def make(cls=BatchedCandidate):
    p=np.column_stack((np.arange(200)*.025,np.zeros(200),np.ones(200)))
    return cls(0,frozenset([2]),2,0,p,np.zeros((200,3)))


def observe(c, frame, indices, code=CONTRADICTION, x=None):
    codes=np.full(200,UNKNOWN,np.uint8);codes[indices]=code
    src=np.where(codes==CONTRADICTION,SOURCE_DEPTH,0).astype(np.uint8)
    pose=np.eye(4);pose[0,3]=frame*.01 if x is None else x
    return c.observe(frame,pose,codes,src,CFG)


class TestBatchedEvidence(unittest.TestCase):
    def test_small_views_can_form_second_valid_packet(self):
        c=make()
        observe(c,0,slice(0,100))
        observe(c,5,slice(0,40))
        observe(c,10,slice(40,80))
        self.assertEqual(c.state,'suspected')
        observe(c,15,slice(80,100))
        self.assertEqual(c.state,'would_remove')
        self.assertEqual(c.packet_events[-1]['distinct_points'],100)
        self.assertEqual(c.packet_events[-1]['distinct_source_frames'],3)

    def test_repeated_40_points_do_not_count_as_100(self):
        c=make();observe(c,0,slice(0,100))
        for frame in (5,10,15,20):observe(c,frame,slice(0,40))
        self.assertEqual(c.state,'suspected')
        self.assertEqual(c.metadata()['pending_distinct_points'],40)

    def test_latest_support_overwrites_old_conflict_locally(self):
        c=make();observe(c,0,slice(0,100))
        observe(c,5,slice(0,60))
        observe(c,10,slice(0,60),SUPPORT)
        observe(c,15,slice(60,100),SUPPORT)
        self.assertEqual(c.packet_events[-1]['reason'],'support')
        self.assertEqual(c.contradicted.sum(),0)
        self.assertEqual(c.state,'suspected')

    def test_invalid_and_occluded_points_do_not_fill_packet(self):
        c=make();observe(c,0,slice(0,100))
        observe(c,5,slice(0,40))
        observe(c,10,slice(40,200),OCCLUDED)
        observe(c,15,slice(40,200),UNKNOWN)
        self.assertEqual(c.metadata()['pending_distinct_points'],40)

    def test_same_view_does_not_add_new_positions(self):
        c=make();observe(c,0,slice(0,40),x=0)
        observe(c,5,slice(40,100),x=0)
        self.assertEqual(c.metadata()['pending_distinct_points'],40)
        self.assertEqual(c.rejected_correlated_frames,1)

    def test_full_frames_match_existing_local_clear_rule(self):
        a,b=make(),make(LocalClearCandidate)
        for frame,indices,code in [(0,slice(0,100),CONTRADICTION),
                                   (5,slice(100,200),SUPPORT),
                                   (10,slice(0,100),CONTRADICTION),
                                   (15,slice(0,100),CONTRADICTION)]:
            observe(a,frame,indices,code);observe(b,frame,indices,code)
            self.assertEqual(a.state,b.state)
            self.assertEqual(a.hits,b.hits)
            np.testing.assert_array_equal(a.depth_contradicted,b.depth_contradicted)

    def test_too_small_pending_never_becomes_confirmation_by_waiting(self):
        c=make();observe(c,0,slice(0,100));observe(c,5,slice(0,40))
        observe(c,10000,slice(0,200),UNKNOWN)
        self.assertEqual(c.state,'suspected')
        self.assertEqual(c.hits,1)


if __name__=='__main__':unittest.main()
