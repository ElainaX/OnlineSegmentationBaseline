import ast, subprocess
from pathlib import Path
import networkx as nx
import numpy as np
import torch
from submap_disjoint.sparse_edges import Edges, sparse_clusters

def extract(source,name,env):
    tree=ast.parse(source)
    f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[f],type_ignores=[]),'<reference>','exec'),env)
    return env[name]

original=extract(Path('/root/OnlineAnySegBaseline_original/tool/helper_functions.py').read_text(),'do_clustering',dict(np=np,nx=nx))
old=extract(subprocess.check_output(['git','show','b048c2b:submap_disjoint/sparse_edges.py'],text=True),'sparse_clusters',dict(np=np))
rng=np.random.default_rng(27)
def clean(groups): return [g.tolist() for g in groups]
for case in range(250):
    n=int(rng.integers(1,40)); matrix=rng.random((n,n))<rng.uniform(0,.4)
    np.fill_diagonal(matrix,False)
    if case%2: matrix=np.tril(matrix)
    pairs=[tuple(map(int,p)) for p in np.argwhere(matrix)]
    edges=Edges({p:float(rng.random()) for p in pairs})
    audit=dict(merge_calls=0,candidate_edges=0,selected_pairs=0)
    actual=sparse_clusters(pairs,n,edges,audit,mode='baseline_components')
    assert clean(actual)==clean(original(torch.from_numpy(matrix))),case
    x=dict(merge_calls=0,candidate_edges=0,selected_pairs=0); y=x.copy()
    assert clean(sparse_clusters(pairs,n,edges,x))==clean(old(pairs,n,edges,y)),case
    assert x==y,case
audit=dict(merge_calls=0,candidate_edges=0,selected_pairs=0)
assert clean(sparse_clusters([(1,0),(2,1)],4,Edges(),audit,'baseline_components'))==[[0,1,2],[3]]
print('PASS: 250 baseline connected-component equivalence cases; 250 unchanged disjoint cases; bridge chain with isolate.')
