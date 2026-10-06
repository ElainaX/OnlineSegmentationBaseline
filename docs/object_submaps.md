# Mask-driven object submaps

The `submaps` option in `config/3rscan_submaps.yaml` tracks object hypotheses
from the masks that OnlineAnySeg accepts. It does not split a submap from a
camera translation or rotation threshold.

For each accepted mask the tracker stores the frame, raw mask id, merged mask
id, pose, masked RGB-D points, and mask feature. A match first uses 3D point
coverage and feature similarity. The one-to-one assignment rejects ambiguous
matches. If a new view has no shared visible surface with an existing track, a
second re-identification pass compares the object-level feature, the gap
between accumulated 3D extents, and temporal continuity. This covers a
front-to-back view change without using a camera rotation split rule.

When `Scene_rep.maskGraph` links observations, `SubmapManager.graph_update`
records identity evidence instead of immediately unioning unrelated tracks.
Disjoint masks that are independently visible in the same frame are treated as
negative evidence. Cross-view links need repeated frame support before the
lineage union is committed, so a temporary mask-graph error is less likely to
create a permanent submap merge.

The tracker also checks RGB-D history from before the current frame. A point is
marked as a possible physical appearance only when two separated historical
views contain valid depth behind the current surface. Missing or occluded depth
is treated as unknown. `first_observed` is therefore the normal label for an
object first seen in the sequence; it is not a claim that the object physically
entered the scene.

At export, each confirmed object is reconstructed sequentially in its own
local Open3D CUDA TSDF. Only one object TSDF is resident at a time, so memory is
bounded by the configured `max_object_blocks`. The first RGB-D view is rendered
again from that camera and placed in the atlas.

The confirmed TSDF surfaces are then projected back onto the global
reconstruction. Each global point is assigned to the closest object surface
within `global_assign_distance_m`; overlapping hypotheses keep only the closest
label. This writes the normal `ckpt_final.npz` format, while the original
Scene_rep-only checkpoint is retained as `ckpt_scene_rep_final.npz` for AP
comparison. The global geometry in `final.ply` is unchanged.

For a run with output directory `<OUT>/<SEQ>`, the main artifacts are:

- `new_objects_first_view_atlas.png`: first RGB+mask and same-camera TSDF
  render for every confirmed object, annotated with the first observed frame.
- `new_object_tracks.json`: observations, merge lineage, graph snapshots,
  history evidence, and export statistics.
- `global_submap_instances.json`: the point assignment threshold and counts
  used to create the global instance checkpoint.
- `objects/object_XXXX/`: per-object first-view images, metadata, and TSDF
  surface/volume files.

The tracker does not use ground-truth instances, class labels, or a second
segmentation model. Ground truth should be used separately for evaluation.
