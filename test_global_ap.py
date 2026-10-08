import unittest
import numpy as np
from eval_global_ap import aggregate_iou,score_sets,combine,retention,deletion


class GlobalAPTests(unittest.TestCase):
    def test_perfect_retention_and_deletion(self):
        a=aggregate_iou(np.eye(2),[10,20],[10,20])
        self.assertEqual(a['score'],1);self.assertEqual(combine(a,a),1)

    def test_extra_prediction_is_not_filtered(self):
        a=aggregate_iou(np.array([[1.],[0.]]),[10],[10,5])
        self.assertAlmostEqual(a['score'],10/15)
        self.assertEqual(a['extra_predictions'],1)

    def test_missing_GT_zero_and_no_GT_not_perfect(self):
        self.assertEqual(aggregate_iou(np.empty((0,1)),[10],[])['score'],0)
        self.assertEqual(aggregate_iou(np.empty((1,0)),[],[10])['score'],0)
        self.assertIsNone(aggregate_iou(np.empty((0,0)),[],[])['score'])

    def test_one_prediction_cannot_cover_two_GT_for_free(self):
        a=aggregate_iou([[.5,.5]],[10,10],[20])
        self.assertEqual(sum(r['prediction'] is not None for r in a['per_GT']),1)
        self.assertEqual(a['score'],.25)

    def test_weighted_iou_not_detection_AP(self):
        a=aggregate_iou([[.5,0],[0,1]],[10,30],[10,30])
        self.assertEqual(a['score'],.875)

    def test_deletion_example_70_correct_10_wrong(self):
        gt=np.r_[np.ones(100,int),np.full(10,-1)]
        pred=np.zeros((1,110),bool);pred[0,:70]=True;pred[0,100:]=True
        a=score_sets(gt,[1],pred,[0]);self.assertAlmostEqual(a['score'],70/110)

    def test_off_GT_surface_in_same_prediction_enters_union(self):
        gt=np.array([[0.,0,0],[.1,0,0]])
        p=np.vstack([gt,[5.,0,0]])
        a=retention(p,np.ones((3,1),bool),gt,np.ones(2,int),.025,.05)
        self.assertAlmostEqual(a['score'],2/3)

    def test_completely_shadowed_prediction_is_still_false_positive(self):
        p=np.array([[0.,0,0],[.03,0,0]])
        a=retention(p,np.eye(2,dtype=bool),p[:1],np.ones(1,int),.025,.05)
        self.assertEqual(a['extra_predictions'],1)
        self.assertEqual(a['score'],.5)

    def test_no_deletion_and_full_deletion_and_unbuilt_exclusion(self):
        initial=np.array([[0.,0,0],[.025,0,0],[.05,0,0]])
        gt=np.vstack([initial,[3.,0,0]]);lab=np.full(4,3001)
        finalgt=np.array([[8.,0,0]])
        before=deletion(initial,initial,gt,lab,finalgt,{1},set(),.025,.01)
        after=deletion(initial,np.empty((0,3)),gt,lab,finalgt,{1},set(),.025,.01)
        self.assertEqual(before['score'],0);self.assertEqual(after['score'],1)
        self.assertEqual(after['GT_weight'],3)

    def test_moving_old_surface_is_neutral_and_current_reuse_not_deleted(self):
        initial=np.array([[0.,0,0],[2.,0,0]])
        a=deletion(initial,np.empty((0,3)),initial,np.array([3001,3002]),initial[:1],{1},{2},.025,.05)
        self.assertEqual(a['GT_weight'],0);self.assertEqual(a['scored_deleted_samples'],0)

    def test_geometry_deleted_but_mask_removed_only_is_not_success(self):
        initial=np.array([[0.,0,0]])
        a=deletion(initial,initial,initial,np.array([3001]),np.array([[8.,0,0]]),{1},set(),.025,.05)
        self.assertEqual(a['score'],0)


if __name__=='__main__':unittest.main()
