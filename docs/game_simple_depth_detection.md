# GaME-style Simple TSDF Depth Change Detection

This extension detects potentially stale OnlineAnySeg instances without
modifying the TSDF or deleting masks. It is intended as a safe first stage for
debugging long-term scene changes.

## Processing order

For selected mapping frames, the extension runs before the current RGB-D frame
is integrated:

1. Ray-cast the existing Open3D `VoxelBlockGrid` from the current camera pose.
2. Mark a disappearance conflict where both depths are valid and
   `observed_depth - rendered_depth > depth_threshold`.
3. Back-project the old rendered surface, quantize it to OnlineAnySeg voxels,
   and query the existing voxel-to-mask hash table.
4. Require minimum conflict size and visible-instance coverage.
5. Accumulate only observations separated in time or viewpoint. Three
   independent observations change an instance from `ACTIVE` through
   `SUSPECT` to `STALE` by default.
6. Write an event directory with depth images, gray conflict overlays, an
   instance outline, historical mask views, a point cloud, and JSON metadata.

The sign of the depth difference is important. A closer observation usually
indicates new foreground geometry or occlusion and is not disappearance
evidence.

## Run

Use the change-detection config in place of the original 3RScan config:

```bash
python main.py \
  -c config/3rscan_dynamic_game_simple_depth.yaml \
  --seq_name "$SEQ" \
  -d "$DATA_ROOT/$SEQ" \
  -i "$MASK_ROOT/$SEQ" \
  -o "$OUT_ROOT" \
  --device cuda:0
```

Other dataset configs can enable the same plugin by copying the
`change_detection` block. When the block is absent, or `enabled: false`, the
baseline behavior is unchanged.

## Output

Events are stored under:

```text
<output>/<sequence>/change_debug/
  event_<id>_frame_<frame>_track_<stable_uid>/
    summary.json
    current_rgb.png
    observed_depth.png
    tsdf_rendered_depth.png
    signed_depth_residual.png
    current_conflict_gray.png
    current_instance_overlay.png
    conflict_points.ply
    contact_sheet.png
    history/*.png
```

The `track_uid` is stable when merged-mask indices are compacted. When several
instances merge, the oldest parent UID is retained and their evidence histories
are combined.

## Initial 3RScan thresholds

The supplied config starts with a 5 cm depth threshold, which is twice the
default 2.5 cm OnlineAnySeg voxel size. These values are starting points, not
dataset-independent constants. Check the residual images before lowering the
threshold.

Important controls:

- `interval`: detection frequency in input-frame IDs. Detection is also limited
  by `mapping.keyframe_freq` because it runs at mapping frames.
- `min_map_weight`: minimum TSDF ray-casting weight.
- `min_conflict_voxels` and `min_conflict_ratio`: spatial support required for
  one observation.
- `confirm_hits`: independent observations required for `STALE`.
- `min_hit_frame_gap`, `min_hit_translation`, and `min_hit_rotation_deg`:
  independence constraints.
- `debug.enabled`: disable image/point-cloud output while keeping detection.

## Current limitation

`STALE` is diagnostic state only. The plugin does not call
`Scene_rep.remove_masks()` and does not reset TSDF weights. This prevents an
early false positive from irreversibly damaging the reconstruction. Local TSDF
block reset and recent-keyframe re-fusion should be implemented as a separate,
explicit second stage after the detector has been validated.

