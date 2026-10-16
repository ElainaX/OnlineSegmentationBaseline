"""CPU protocol tests: original AP reduction, action audit, safe aggregation."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import pickle

import numpy as np
import gap_core as g
import metrics as m


def fixture(seq, detected=True, score=.9, n_gt=1):
    labels = m.reducer()['CLASS_LABELS']
    label = labels[0]
    gt = {k: [] for k in labels}; pred = {k: [] for k in labels}
    for i in range(n_gt):
        gt[label].append(dict(instance_id=3001+i, vert_count=100, med_dist=0.,
                              dist_conf=1., matched_pred=[]))
    if detected:
        p = dict(filename=seq+'.0', pred_id=0, vert_count=100, confidence=score,
                 void_intersection=0, matched_gt=[])
        pg = {k: v for k, v in gt[label][0].items() if k != 'matched_pred'}
        pg['intersection'] = 100
        p['matched_gt'] = [pg]
        gp = {k: v for k, v in p.items() if k != 'matched_gt'}
        gp['intersection'] = 100
        gt[label][0]['matched_pred'] = [gp]
        pred[label].append(p)
    return {seq: dict(gt=gt, pred=pred)}


class MetricsTests(unittest.TestCase):
    def test_gap_evidence_and_boundaries(self):
        g.tests()
        # A surviving neighbour is required to reconstruct a crossing at an erased node.
        z = dict(xyz=np.array([[0.,0,0],[g.V,0,0]]), tsdf=np.array([-.5,.5]),
                 weight=np.ones(2), erased=np.array([True,False]))
        self.assertEqual(len(g.archived_surface(z)[0]), 1)
        z['erased'][:] = False
        self.assertEqual(len(g.archived_surface(z)[0]), 0)
        t = np.array([True,False]); zero = np.zeros(2, bool)
        self.assertIsNone(g.set_score(t,zero,zero)['precision'])
        self.assertIsNone(g.set_score(zero,zero,zero)['recall'])
        self.assertEqual(g.set_score(t,t,zero,extras=1)['D'], .5)

    def test_composites_and_nulls(self):
        r = m.composite(dict(AP=.2, AP50=.4, AP25=.6), .25)
        self.assertAlmostEqual(r['GAP_v3'], .4)
        self.assertAlmostEqual(r['GAP50'], .55)
        self.assertAlmostEqual(r['GAP25'], .7)
        self.assertIsNone(m.composite(dict(AP=None,AP50=None,AP25=None),1)['GAP_v3'])
        self.assertEqual(m.composite(dict(AP=0,AP50=1,AP25=.5),1)['GAP_v3'],1)

    def test_relocated_actions_keep_candidate_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=g.relocated_archive(tmp,'/old/run/correction/actions/12/50.npz')
            b=g.relocated_archive(tmp,'/old/run/correction/actions/13/50.npz')
            self.assertNotEqual(a,b)
            self.assertEqual(a,Path(tmp).resolve()/'actions/12/50.npz')
            with self.assertRaises(ValueError):
                g.relocated_archive(tmp,'/old/run/correction/../../outside.npz')

    def test_archive_audit_same_frame_two_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); actions=[]
            for candidate,offset in [(12,0.),(13,1.)]:
                folder=root/'correction/actions'/str(candidate); folder.mkdir(parents=True)
                z=dict(xyz=np.array([[offset,0,0],[offset+g.V,0,0]]),
                       tsdf=np.array([-.5,.5]),weight=np.ones(2),erased=np.array([True,False]))
                np.savez(folder/'10.npz',**z)
                np.savez(folder/'10.evidence.npz',**z)
                actions.append(dict(frame=10,archive=f'/old/run/correction/actions/{candidate}/10.npz'))
            m.write(root/'correction/summary.json',dict(actions=actions))
            ref=np.array([[g.V/2,0,0],[1+g.V/2,0,0]])
            result=g.evaluate_method(root,ref,np.floor(ref/g.V),np.array([1,2]),
                    np.array([True,False]),{1},set(),np.zeros(2,bool),5,
                    np.array([[10.,10.,10.]]),archive_root=root/'correction')
            self.assertEqual((result['TP'],result['FP'],result['FN']),(1,1,0))
            self.assertEqual(result['D'],.5)
            self.assertEqual(len(result['archive_sha256']),4)

    def test_original_ap_and_pool_not_mean(self):
        a = fixture('a'); b = fixture('b', False, n_gt=3)
        self.assertEqual(m.ap_from_matches(a)['AP'], 1.)
        self.assertEqual(m.ap_from_matches(b)['AP'], 0.)
        pooled = m.ap_from_matches(dict(a, **b))['AP']
        self.assertAlmostEqual(pooled, .25)
        self.assertNotEqual(pooled, .5)
        empty = fixture('empty', False, n_gt=0)
        self.assertIsNone(m.ap_from_matches(empty)['AP'])

    def test_strict_iou_threshold(self):
        data=fixture('half'); label=m.reducer()['CLASS_LABELS'][0]
        # gt=100, pred=200, intersection=100 -> exactly IoU 0.50, not >0.50.
        p=data['half']['pred'][label][0]; p['vert_count']=200
        data['half']['gt'][label][0]['matched_pred'][0]['vert_count']=200
        score=m.ap_from_matches(data)
        self.assertEqual(score['AP50'],0.)
        self.assertEqual(score['AP25'],1.)

    def test_manifest_rejects_duplicate_and_wrong_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'a'; d.mkdir()
            with (d/'ap_matches.pkl').open('wb') as f: pickle.dump(fixture('a'),f)
            r=dict(protocol=m.PROTOCOL,seq='a',method='current',ap_evaluator_sha256=m.sha(m.EVAL),
                   gap_core_sha256=m.sha(m.HERE/'gap_core.py'),matches_sha256=m.sha(d/'ap_matches.pkl'),
                   metrics=m.composite(dict(AP=1.,AP50=1.,AP25=1.),0.),
                   deletion=dict(precision=None,recall=0.),domain_sha256='fixture')
            m.write(d/'metrics.json',r)
            self.assertEqual(m.pool_directories([d])['completed'],1)
            with self.assertRaises(AssertionError): m.pool_directories([d,d])
            r['protocol']='legacy'; m.write(d/'metrics.json',r)
            with self.assertRaises(AssertionError): m.pool_directories([d])

    def test_validation_selection(self):
        rows=m.read(m.HERE/'validation_pairs.json')['rows']
        self.assertEqual(len(rows),110)
        self.assertEqual(len({r['reference'] for r in rows}),47)
        self.assertEqual(len({r['seq'] for r in rows}),110)
        self.assertTrue(all(r['split']=='validation' for r in rows))


if __name__ == '__main__':
    unittest.main()
