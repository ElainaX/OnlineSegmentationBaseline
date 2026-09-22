# GaME-style TSDF Multi-view Change Verification

This plugin detects potentially stale OnlineAnySeg instances without deleting
TSDF voxels or masks. It replaces the former repeated-hit heuristic with a
GaME-style current-view conflict test followed by historical multi-view
verification.

## Processing order

For a selected mapping frame, the plugin runs before that RGB-D frame is fused:

1. Ray-cast depth and color from the existing Open3D `VoxelBlockGrid`.
2. Select pixels where the observation is farther than the old TSDF surface by
   `depth_threshold` **and** the rendered/observed RGB error exceeds
   `color_error_threshold`.
3. Associate the old surface samples with OnlineAnySeg instances through the
   voxel-to-mask hash table.
4. For each candidate, project its conflicting TSDF samples into stored
   historical keyframes that contributed original 2D masks to that instance.
5. In each view, require agreement with historical depth, historical RGB, and
   the original CropFormer mask. A view also has to pass projected-support and
   mask-coverage thresholds.
6. Mark the track `STALE` only when at least `min_verified_views` historical
   keyframes pass. Repeated current-frame hits are no longer accumulated.
7. Store the verified historical mask regions as per-keyframe occlusion masks
   for the later TSDF repair/re-fusion stage.

The current frame is inserted into the bounded CPU keyframe bank only after its
detection step, so it cannot validate itself. Keyframe selection uses frame
spacing plus camera translation/rotation, and therefore does not depend on
`session_boundary.json`.

## Run

The existing config filename is retained so previous commands continue to
work, but its detector is now the multi-view implementation:

```bash
python main.py \
  -c config/3rscan_dynamic_game_simple_depth.yaml \
  --seq_name "$SEQ" \
  -d "$DATA_ROOT/$SEQ" \
  -i "$MASK_ROOT/$SEQ" \
  -o "$OUT_ROOT" \
  --device cuda:0
```

When `change_detection.enabled` is absent or false, the original OnlineAnySeg
pipeline is unchanged.

## Important controls

- `interval`: current-frame conflict detection frequency. Keyframe collection
  can still occur on every mapping frame.
- `depth_threshold`, `color_error_threshold`: GaME-style current-view depth and
  color disagreement gates.
- `min_map_weight`: minimum TSDF weight accepted by Open3D ray casting; this is
  the TSDF counterpart of GaME's opacity gate.
- `keyframe_*`, `max_keyframes`: motion-based CPU keyframe-bank policy.
- `min_verified_views`: number of historical views required to accept a stale
  track.
- `history_depth_tolerance`, `history_color_tolerance`: agreement thresholds in
  a historical view.
- `history_projected_agreement`: fraction of projected candidate samples that
  must agree with depth, color, and the source mask.
- `history_mask_coverage`: fraction of the source CropFormer mask covered by
  agreeing candidate samples. GaME also uses a deliberately low mask-coverage
  threshold to grow a partial conflict to an object mask.
- `debug.preview_interval`: periodic current-view diagnostics even when no
  instance is accepted.

## Output

Outputs remain under `<output>/<sequence>/change_debug/`. Accepted event
directories contain current depth/color conflict views, `conflict_points.ply`,
historical views annotated as `VERIFIED` or `REJECTED`, a contact sheet, and a
`summary.json` containing per-view depth/color/mask agreement values.

Console messages distinguish a rejected current candidate from an accepted
multi-view event:

```text
... history 0/2 -> REJECTED
... history 3/2 -> STALE
```

## Current boundary

This change implements GaME's multi-view verification and keyframe occlusion
state, but not its map mutation step. A `STALE` result therefore does not yet
erase a TSDF surface. Stable voxel deactivation and re-fusion must be added as a
separate repair stage; treating this detector as if it already repaired the
map would be incorrect.
