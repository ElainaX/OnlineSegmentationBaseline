# IG confirmed local deletion experiment

Base: `94e4dde`, IG readonly detector and GT audit. The detector source and
configuration are unchanged. New branch: `codex/ig-confirmed-local-deletion-20261008`.

## Intervention

The standard onlineSeg detector runs before fusion. The deletion controller
runs immediately after it, before current RGB-D integration. A skipped detection
frame (invalid pose/depth, global anomaly, insufficient render) cannot erase.

For each confirmed `would_remove` candidate:

1. Use only the frozen reference points with accumulated **depth** contradiction.
   RGB-only change never authorizes geometry erasure. The detector confirms two
   independent observations at instance level; this does NOT mean every point
   individually had two contradiction votes.
2. Generate the eight existing TSDF lattice corners of those surface samples.
   Do not expand to the entire AABB, entire track, or truncation band.
3. Check every corner against current raw depth, respecting depth holes, depth
   boundaries, valid range, camera pose and the original absolute/relative
   epsilon. Only current free-space evidence can pass. Occlusion cannot pass.
4. Require nonempty active raw-mask memberships and require all of them to belong
   to the candidate's frozen raw lineage. A later mask sharing the current merged
   slot is not permission to erase it. This is deliberately conservative.
5. Archive old numeric attributes and memberships, then zero TSDF, weight and
   color at eligible corners. No hashmap blocks are deallocated.
6. Remove those corners from live instances, original per-frame geometry caches,
   memberships and voxel visibility. Remove empty tracks through the existing
   sparse owner's compaction function. Recompute sparse overlap edges and
   invalidate FCGF/point/segmentation caches. Existing semantic embeddings and
   observation counts remain historical; no unsupported inverse pooling is used.

Confirmed candidates remain eligible in later valid views, but their frozen
depth-contradicted region does not expand after confirmation. Current depth and
raw lineage are checked again. Normal subsequent fusion may refill geometry.

## Controls and evaluation

Four pairs; 25 mm TSDF; disjoint pairs on; Carving-AABB and depth shrink off.
pair01/03/04: mapping and segmentation stride 2. pair02: stride 5.
Each pair gets a new detection-on/deletion-off run and a deletion-on run on the
same GPU and identical input/config/seed. Original evaluator uses the full aligned
rescan GT and original class-agnostic protocol, without excluding changed objects.

The initial pair01 deletion run stopped on an empty Open3D hash lookup after two
successful erasures. A regression test and empty-proposal guard fixed that
boundary case without altering any numerical decision. Its retry uses GPU 1;
the pair01 control used GPU 3. The failed attempt is retained separately.

Data remain unchanged. GT is not accessed by mapping or deletion decisions.
Original baseline, IG readonly and IT branches/results are preserved. Shared-GPU
wall times are descriptive, not a controlled throughput benchmark. Original GPU
pipeline has known run-to-run variation; a single small AP delta is not evidence
of a robust improvement.

## Validation and limits

CUDA tests check true voxel mutation, unchanged neighboring geometry, archived
pre-mutation weights, cleaned memberships/visibility, and coordinate-index
consistency even if the original coordinate/index arrays have different order.
Other tests cover occlusion, holes and new raw observations sharing an old slot.

The original `remove_detection/summary.json` still correctly describes the
unchanged detector as read-only and says zero detector deletions. Actual mutation
counts belong to `deletion/summary.json` and `run_metadata.deletion_summary`.
Do not use the detector's zero counter as the total pipeline deletion count.

This tests **local geometry-and-membership correction**, not wholesale instance
retirement. Archives preserve erasure evidence; they are not a complete automatic
graph rollback. Frozen sampling, corner-only erasure, the lineage gate, surviving
semantic contamination and fragmented detections can all limit AP improvement.
