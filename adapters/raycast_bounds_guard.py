"""Open3D 0.18 range-map allocation guard; no detector or TSDF changes.

Keep the existing ray caster, intrinsics, depth limits, step sizes, attributes,
and default range-map downsampling. Pad only the output extent to a multiple
of the range-map factor, then return the original pixel rectangle.
"""
import functools
import json
from pathlib import Path


class GridView:
    def __init__(self, grid):
        self.grid = grid

    def __getattr__(self, name):
        return getattr(self.grid, name)

    def ray_cast(self, *args, **kwargs):
        args = list(args)
        width = int(args[3] if len(args) > 3 else kwargs['width'])
        height = int(args[4] if len(args) > 4 else kwargs['height'])
        factor = int(args[11] if len(args) > 11 else kwargs.get('range_map_down_factor', 8))
        assert width > 0 and height > 0 and factor > 0
        padded_w = ((width + factor - 1) // factor) * factor
        padded_h = ((height + factor - 1) // factor) * factor
        if (padded_w, padded_h) == (width, height):
            return self.grid.ray_cast(*args, **kwargs)
        if len(args) > 3:
            args[3] = padded_w
        else:
            kwargs['width'] = padded_w
        if len(args) > 4:
            args[4] = padded_h
        else:
            kwargs['height'] = padded_h
        rendered = self.grid.ray_cast(*args, **kwargs)
        return {name: (value if name == 'range' else value[:height, :width].contiguous())
                for name, value in rendered.items()}


class GeometryView:
    def __init__(self, geometry):
        self.geometry = geometry
        self.voxel_block_grids = GridView(geometry.voxel_block_grids)

    def __getattr__(self, name):
        return getattr(self.geometry, name)


def install(output=None):
    import open3d
    import feng_ig_adapter
    assert open3d.__version__ == '0.18.0', open3d.__version__
    original = feng_ig_adapter.GeometryView.raycast_history
    if getattr(original, '_range_bounds_guard', False):
        return

    @functools.wraps(original)
    def guarded(self, *args, **kwargs):
        return original(GeometryView(self), *args, **kwargs)

    guarded._range_bounds_guard = True
    feng_ig_adapter.GeometryView.raycast_history = guarded
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        (output.parent / (output.name + '.raycast_bounds_guard.json')).write_text(json.dumps({
            'open3d': open3d.__version__,
            'fix': 'ceil output extent to range-map factor, crop to original pixels',
            'intrinsics_unchanged': True,
            'detector_thresholds_unchanged': True,
            'repository_files_unchanged': True,
            'default_range_map_down_factor': 8,
            'source': 'Open3D v0.18.0 VoxelBlockGridImpl.h EstimateRange and RayCast',
        }, indent=2))
