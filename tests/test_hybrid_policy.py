import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'adapters')]
import json,unittest
from pathlib import Path
import numpy as np
from feng_remove_detection import Candidate,checked_settings,CONTRADICTION,SUPPORT,OCCLUDED,SOURCE_DEPTH,SOURCE_RGB
from hybrid_confirmation import observe

class HybridTests(unittest.TestCase):
    def setUp(self):
        r=ROOT/'configs'
        self.cfg=json.loads((r/'detector_H.json').read_text())
        checked_settings(self.cfg)
    def candidate(self,n=200):return Candidate(1,frozenset({0}),0,0,np.ones((n,3)),np.ones((n,3)))
    def evidence(self,d=0,rgb=0,support=0,n=200):
        codes=np.full(n,OCCLUDED,np.uint8);src=np.zeros(n,np.uint8)
        codes[:d+rgb]=CONTRADICTION;src[:d]=SOURCE_DEPTH;src[d:d+rgb]=SOURCE_RGB
        codes[d+rgb:d+rgb+support]=SUPPORT
        return codes,src
    def step(self,c,frame,d=0,rgb=0,support=0,translation=None):
        pose=np.eye(4);pose[0,3]=frame*.02 if translation is None else translation
        return observe(Candidate.observe,c,frame,pose,*self.evidence(d,rgb,support,len(c.points)),self.cfg)
    def test_dual_first(self):
        c=self.candidate();state,s=self.step(c,0,60,40)
        self.assertEqual(state,'would_remove');self.assertEqual(c.hits,1)
        self.assertEqual(s['hybrid_effective_confirmations'],1)
    def test_depth_then_depth(self):
        c=self.candidate();self.assertEqual(self.step(c,0,100)[0],'contradiction')
        self.assertEqual(self.step(c,6,100)[0],'would_remove')
    def test_rgb_then_rgb(self):
        c=self.candidate();self.assertEqual(self.step(c,0,rgb=100)[0],'contradiction')
        self.assertEqual(self.step(c,6,rgb=100)[0],'would_remove')
    def test_depth_then_rgb(self):
        c=self.candidate();self.assertEqual(self.step(c,0,100)[0],'contradiction')
        self.assertEqual(self.step(c,6,rgb=100)[0],'would_remove')
    def test_tiny_second_source_does_not_fast_confirm(self):
        self.assertEqual(self.step(self.candidate(),0,99,1)[0],'contradiction')
        self.assertEqual(self.step(self.candidate(),0,61,39)[0],'contradiction')
    def test_dual_below_total_ratio(self):
        c=self.candidate();self.assertEqual(self.step(c,0,40,40,100)[0],'paused_mixed_evidence')
        self.assertEqual(c.hits,0)
    def test_dual_below_tested_point_minimum(self):
        c=self.candidate();self.assertEqual(self.step(c,0,40,40)[0],'paused_insufficient_reference_points')
    def test_motion_and_gap_not_bypassed(self):
        c=self.candidate();self.step(c,0,100)
        self.assertEqual(self.step(c,1,60,40)[0],'paused_correlated_observation')
        self.assertEqual(self.step(c,6,60,40,translation=0)[0],'paused_correlated_observation')
        self.assertEqual(c.hits,1)
    def test_reference_coverage_not_bypassed(self):
        c=self.candidate(1000)
        self.assertEqual(self.step(c,0,100)[0],'contradiction')
        self.assertEqual(self.step(c,6,100)[0],'contradiction')
    def test_support_resets_and_new_episode(self):
        c=self.candidate();self.step(c,0,100)
        self.assertEqual(self.step(c,6,support=200)[0],'support');self.assertEqual(c.hits,0)
        self.assertEqual(self.step(c,12,60,40)[0],'would_remove')
    def test_support_cancels(self):
        c=self.candidate();self.step(c,0,100);self.step(c,6,support=200)
        self.assertEqual(self.step(c,12,support=200)[0],'cancelled')
        self.assertEqual(self.step(c,18,60,40)[0],'terminal')
    def test_single_source_matches_base_state_machine(self):
        a,b=self.candidate(),self.candidate()
        for frame,d,support in [(0,100,0),(1,100,0),(6,0,200),(12,100,0),(18,100,0)]:
            pose=np.eye(4);pose[0,3]=frame*.02
            codes,src=self.evidence(d,0,support)
            expected=Candidate.observe(a,frame,pose,codes,src,self.cfg)[0]
            actual=observe(Candidate.observe,b,frame,pose,codes,src,self.cfg)[0]
            self.assertEqual(expected,actual);self.assertEqual(a.hits,b.hits)
            np.testing.assert_array_equal(a.contradicted,b.contradicted)

if __name__=='__main__':unittest.main()
