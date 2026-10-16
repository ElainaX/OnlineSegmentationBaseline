"""CPU equivalence tests; fake TSDF isolates observation/score cache behavior."""
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from submap_disjoint import online_fine_evidence as fast
from submap_disjoint.multires_policy import project_mask_points

original_path = '/root/OnlineSegmentationBaseline_multires_preserve_coarse_20261013/submap_disjoint/online_fine_evidence.py'
spec = importlib.util.spec_from_file_location('submap_disjoint.original_online_fine', original_path)
original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)


class FakeVolume:
    def __init__(self, options):
        self.options = options; self.points = []
    def integrate(self, rgb, depth, K, pose, mask):
        self.points.append(project_mask_points(depth, pose, K, mask, .1, 5., 1200))
    def surface(self):
        pts = np.concatenate(self.points) if self.points else np.empty((0, 3))
        return pts, np.zeros_like(pts)


def cfg():
    return dict(scene=dict(block_resolution=4), cam=dict(depth_near=.1, depth_far=5.),
        multi_resolution=dict(fine_voxel_size_m=.015, physical_truncation_m=.1,
                              fine_min_views=3, fine_max_views=5, fine_max_extent_m=.45))


def owner(K):
    return SimpleNamespace(intrinsics=torch.tensor(K), device='cuda:0',
                           global2merged_mask_id=[], mask_voxels_coords={})


def run_pair(packets, mappings, K, destination):
    owners = [owner(K), owner(K)]
    engines = [mod.OnlineFineEvidence(cfg(), own, Path(destination)/str(i))
               for i, (mod, own) in enumerate(zip([original, fast], owners))]
    timings = [[], []]
    for end, mapping in mappings:
        for i, (engine, own) in enumerate(zip(engines, owners)):
            own.global2merged_mask_id = list(mapping)
            start = time.perf_counter(); engine.refresh(packets[:end]); timings[i].append(time.perf_counter()-start)
        assert engines[0].cache.keys() == engines[1].cache.keys()
        for key in engines[0].cache:
            np.testing.assert_array_equal(engines[0].cache[key], engines[1].cache[key])
        for key in ['updates', 'fine_builds', 'skipped_block_budget', 'last_decisions']:
            assert engines[0].audit[key] == engines[1].audit[key], key
    return engines, timings


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.patches = [patch.object(torch.cuda, 'synchronize', lambda: None),
                        patch.object(fast.o3c.cuda, 'synchronize', lambda: None)]
        for mod in [original, fast]:
            self.patches += [patch.object(mod, 'LocalTSDF', FakeVolume),
                             patch.object(mod, 'reserve_window', lambda *args: None)]
        for p in self.patches: p.start()
        self.temp = tempfile.TemporaryDirectory()
    def tearDown(self):
        self.temp.cleanup()
        for p in reversed(self.patches): p.stop()

    def test_merges_removal_window_changes_and_unchanged_tracks(self):
        K = np.array([[100., 0, 16.], [0, 100., 16.], [0, 0, 1.]])
        packets, mappings, mapping = [], [], []
        for f in range(10):
            labels = np.zeros((32, 32), np.int32); labels[8:20, 8:20] = 1
            # Track 1 stops receiving observations; cache must reuse decisions.
            pairs = [(1, len(mapping))]; mapping.append(0)
            if f < 2:
                labels[22:30, 22:30] = 2; pairs.append((2, len(mapping))); mapping.append(1)
            depth = np.ones((32, 32), np.float32)
            if f == 6: depth[8:20, 8:20] = 2.5
            packets.append(dict(frame=f, labels=labels, depth=depth, rgb=np.zeros((32,32,3),np.uint8), pose=np.eye(4), pairs=pairs))
            if f == 5: mapping = [0 if x == 1 else x for x in mapping]
            if f == 8: mapping[0] = -1
            mappings.append((len(packets), list(mapping)))
        engines, _ = run_pair(packets, mappings, K, self.temp.name)
        self.assertGreater(engines[1].audit['resolution_cache_hits'], 0)

    def test_duplicate_frames_and_multiple_raw_masks_in_a_track(self):
        K = np.diag([100., 100., 1.]); packets=[]
        for i, frame in enumerate([0, 1, 1, 2, 3, 4]):
            labels=np.ones((16,16),np.int32);labels[:8]=2
            packets.append(dict(frame=frame,labels=labels,depth=np.ones((16,16)),rgb=np.zeros((16,16,3),np.uint8),pose=np.eye(4),pairs=[(1,2*i),(2,2*i+1)]))
        run_pair(packets, [(i, [0]*(2*i)) for i in range(1,7)], K, self.temp.name)

    def test_score_values_identical_with_repeated_endpoints(self):
        K=np.array([[100.,0,32.],[0,100.,32.],[0,0,1.]])
        rng=np.random.default_rng(5);points=rng.uniform(-.1,.1,(200,3));points[:,2]+=1.
        engines=[]
        for i,mod in enumerate([original,fast]):
            own=owner(K);own.global2merged_mask_id=[0,1,2]
            own.mask_voxels_coords={j:torch.tensor(points+[j*.005,0,0]) for j in range(3)}
            engine=mod.OnlineFineEvidence(cfg(),own,Path(self.temp.name)/str(i))
            engine.cache={(0,):points,(1,):points+.001}
            engine.packet=dict(pose=np.eye(4),depth=np.ones((64,64)))
            engines.append(engine)
        pairs=torch.tensor([[0,1],[0,2],[1,2],[0,1]]*8);values=torch.linspace(.1,.8,len(pairs))
        a=engines[0].adjust_candidate_scores(values,pairs,None,None)
        b=engines[1].adjust_candidate_scores(values,pairs,None,None)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(engines[1].audit['score_trees_built'],3)
        self.assertEqual(engines[1].audit['depth_evidence_evaluations'],2)


def benchmark():
    import cv2
    root=Path('/root/autodl-tmp/multires_speed_diagnosis_20261013');root.mkdir(exist_ok=True)
    manifests=json.load(open('/root/autodl-tmp/multires_corrected7_20261013_v2/manifest.json'))
    result={}
    test=CacheTests();test.setUp()
    try:
        for seq,limit in [('scene0050_00',425),('scene0608_00',305)]:
            m=next(x for x in manifests if x['seq']==seq)
            migration=json.load(open('/root/autodl-tmp/ig_global25_ten_20261013/IG_global25/'+seq+'/submap_migration.json'))
            lookup={tuple(obs):inst['id'] for inst in migration['instances'] for obs in inst['observations']}
            frames=json.load(open(m['frame_ids_file']))[:limit]
            K=np.loadtxt(Path(m['dataset'])/'intrinsic/intrinsic_depth.txt')[:3,:3].astype(np.float32).astype(np.float64)
            K[0,2]-=10;K[1,2]-=10
            packets=[];mapping=[];mappings=[]
            for frame in frames:
                labels=cv2.imread(str(Path(m['masks'])/'mask'/('%d.png'%frame)),cv2.IMREAD_UNCHANGED)[10:-10,10:-10].astype(np.int32)
                depth=cv2.imread(str(Path(m['dataset'])/'depth'/('%d.png'%frame)),cv2.IMREAD_UNCHANGED)[10:-10,10:-10].astype(np.float32)/1000.
                depth=np.where((depth>.1)&(depth<5.),depth,0)
                pose=np.loadtxt(Path(m['dataset'])/'pose'/('%d.txt'%frame)).astype(np.float32).astype(np.float64)
                pairs=[]
                for raw in np.unique(labels):
                    if (frame,int(raw)) in lookup:
                        pairs.append((int(raw),len(mapping)));mapping.append(lookup[(frame,int(raw))])
                packets.append(dict(frame=frame,labels=labels,depth=depth,pose=pose,rgb=None,pairs=pairs))
                if len(packets)>=len(frames)-10:mappings.append((len(packets),list(mapping)))
            engines,times=run_pair(packets,mappings,K,root/seq)
            result[seq]=dict(note='CPU refresh only; real RGB-D masks with frozen old final associations; TSDF mocked identically. Not an end-to-end runtime prediction.',
                snapshots=len(mappings),original_seconds=times[0],optimized_seconds=times[1],
                original_total_seconds=sum(times[0]),optimized_total_seconds=sum(times[1]),
                speedup=sum(times[0])/sum(times[1]),cache_and_decisions_exactly_equal=True,
                new_cache_hits=engines[1].audit['resolution_cache_hits'],materialized_masks=engines[1].audit['materialized_masks'])
            (root/'cpu_cache_benchmark.json').write_text(json.dumps(result,indent=2))
            print(seq,json.dumps(result[seq]),flush=True)
    finally:
        test.tearDown()


if __name__=='__main__':
    torch.set_num_threads(1)
    if sys.argv[1:]==['benchmark']:benchmark()
    else:unittest.main(verbosity=2)
