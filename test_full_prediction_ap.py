import unittest
import numpy as np
from eval_full_prediction_ap import strict_ap, projected_iou, voxel_union_iou


class FullPredictionTests(unittest.TestCase):
    def test_extra_object_counts_as_fp_and_removal_improves(self):
        before=strict_ap(np.array([[1.],[0.]]),np.ones(2))
        after=strict_ap(np.array([[1.]]),np.ones(1))
        self.assertAlmostEqual(before['AP50'],.75)
        self.assertAlmostEqual(after['AP50'],1.)
        self.assertEqual(before['thresholds'][1]['FP'],1)

    def test_delete_true_object_hurts_and_delete_all_is_not_perfect(self):
        full=strict_ap(np.eye(2),np.ones(2))
        partial=strict_ap(np.array([[1.,0.]]),np.ones(1))
        empty=strict_ap(np.empty((0,2)),np.empty(0))
        self.assertLess(partial['AP50'],full['AP50'])
        self.assertEqual(empty['AP50'],0.)

    def test_duplicate_and_ambiguous_pair_are_one_to_one(self):
        duplicate=strict_ap(np.array([[1.],[1.]]),np.ones(2))
        self.assertEqual(duplicate['thresholds'][1]['TP'],1)
        self.assertEqual(duplicate['thresholds'][1]['FP'],1)
        matrix=np.array([[.8,.8],[.8,0.]])
        self.assertEqual(strict_ap(matrix,np.ones(2))['AP50'],1.)
        self.assertEqual(strict_ap(matrix[::-1],np.ones(2))['AP50'],1.)

    def test_score_order_and_tied_groups(self):
        iou=np.array([[1.],[0.]])
        good=strict_ap(iou,np.array([.9,.1]))['AP50']
        bad=strict_ap(iou,np.array([.1,.9]))['AP50']
        self.assertGreater(good,bad)
        self.assertEqual(strict_ap(iou,np.ones(2))['AP50'],strict_ap(iou[::-1],np.ones(2))['AP50'])

    def test_empty_gt_reports_false_positives_not_perfect_score(self):
        result=strict_ap(np.empty((2,0)),np.ones(2))
        self.assertIsNone(result['AP50'])
        self.assertEqual(result['thresholds'][1]['FP'],2)

    def test_projection_keeps_entirely_off_gt_object(self):
        p=np.array([[0.,0.,0.],[10.,0.,0.]])
        masks=np.eye(2,dtype=bool);labels=np.array([1001]);ids=np.array([1001])
        matrix,sizes=projected_iou(p,masks,p[:1],labels,ids,.15)
        np.testing.assert_array_equal(matrix,[[1.],[0.]])
        np.testing.assert_array_equal(sizes,[1,0])
        self.assertEqual(strict_ap(matrix,np.ones(2))['thresholds'][1]['FP'],1)

    def test_extra_surface_in_same_instance_reduces_union_iou(self):
        p=np.array([[.01,0.,0.],[1.,0.,0.]])
        masks=np.ones((2,1),dtype=bool);labels=np.array([1001]);ids=np.array([1001])
        matrix,_,_=voxel_union_iou(p,masks,p[:1],labels,ids,.025)
        self.assertEqual(matrix[0,0],.5)
        cleaned,_,_=voxel_union_iou(p[:1],masks[:1],p[:1],labels,ids,.025)
        self.assertEqual(cleaned[0,0],1.)

    def test_voxel_density_invariance_and_negative_coordinates(self):
        p=np.array([[-.01,0.,0.],[.03,0.,0.]])
        g=p.copy();labels=np.array([1001,1001]);ids=np.array([1001])
        matrix,_,_=voxel_union_iou(np.repeat(p,5,axis=0),np.ones((10,1),bool),g,labels,ids,.025)
        self.assertEqual(matrix[0,0],1.)

    def test_invalid_score_fails(self):
        with self.assertRaises(ValueError): strict_ap(np.ones((1,1)),[np.nan])

    def test_f1_penalizes_late_false_positive_even_if_ap_saturated(self):
        with_ghost=strict_ap(np.array([[1.],[0.]]),np.array([.9,.1]))
        without_ghost=strict_ap(np.array([[1.]]),np.array([.9]))
        self.assertEqual(with_ghost['AP50'],without_ghost['AP50'])
        self.assertLess(with_ghost['F1_50'],without_ghost['F1_50'])


if __name__=='__main__': unittest.main()
