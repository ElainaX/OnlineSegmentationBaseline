"""External no-valid-depth guard; original repository files remain unchanged."""
import ast
import hashlib

def run_guarded(path):
    source=path.read_bytes()
    tree=ast.parse(source,filename=str(path))
    class Guard(ast.NodeTransformer):
        count=0
        def visit_Assign(self,node):
            call=node.value
            if not (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                    and call.func.attr=='integrate_frame'
                    and isinstance(call.func.value,ast.Name) and call.func.value.id=='scene_rep'):
                return node
            self.count+=1
            handler=ast.parse('''
try:
    pass
except RuntimeError as error:
    if 'No block is touched in TSDF volume' not in str(error):
        raise
    valid_depth = (depth_img[0] > cfg['cam']['depth_near']) & (depth_img[0] < cfg['cam']['depth_far'])
    if bool(valid_depth.any().item()):
        raise
    __empty_depth_skipped.append(int(frame_id))
    print('BASELINE_EMPTY_DEPTH_SKIP: frame=' + str(frame_id), flush=True)
    continue
''').body[0]
            handler.body=[node]
            return ast.copy_location(handler,node)
    transform=Guard();tree=transform.visit(tree)
    assert transform.count==1,'Unexpected original entry point; no broad exception suppression'
    ast.fix_missing_locations(tree)
    namespace=dict(__name__='__main__',__file__=str(path),__package__=None,
                   __empty_depth_skipped=[],__original_main_sha256=hashlib.sha256(source).hexdigest())
    exec(compile(tree,str(path),'exec'),namespace)
    return namespace
