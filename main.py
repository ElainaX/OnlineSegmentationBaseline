import numpy as np
import sys
sys.path.append("third_party/FCGF")
import argparse
import os
import shutil
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from Dataset.dataset import get_dataset
import tool.config as config
from Scene_rep import Scene_rep
from submap_mapping import SubmapManager

max_frame_num = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./config/scannet_yuan.yaml")
    parser.add_argument("-d", "--dataset", type=str, default="~/t7/ScanNet/aligned_scans")
    parser.add_argument("-i", "--instance_dir", type=str, default="./data/grounding_sam_result/scannet/scene0653_00")
    parser.add_argument("--seq_name", default="scene0011_00")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("-o", "--output_dir", default="./output/scannet")
    parser.add_argument(
        "--frame_stride", type=int, default=0,
        help="Frame sampling stride. 0 selects an automatic stride from the config.")
    args = parser.parse_args()

    cfg = config.load_config(args.config)
    instance_owner = str(cfg.get("instance_owner", "baseline")).lower()
    if instance_owner == "submap" and not bool(cfg.get("submaps", {}).get("enabled", False)):
        raise ValueError("instance_owner=submap requires submaps.enabled=true")

    dataset = get_dataset(args.dataset, args.instance_dir, cfg, args.device)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    last_seg_frame_id = dataset.last_seg_frame_id

    # Long RGB-D sequences contain thousands of redundant frames while the
    # cached masks arrive at a much lower interval.  Keep short sequences
    # dense, and for long sequences choose a stride aligned to the mask
    # interval so every processed segmentation frame still has its embedding.
    configured_limit = int(cfg.get('experiment', {}).get('max_processed_frames', 600))
    if args.frame_stride > 0:
        frame_stride = int(args.frame_stride)
    elif configured_limit > 0 and len(dataset) > configured_limit:
        raw_stride = int((len(dataset) + configured_limit - 1) / configured_limit)
        mask_interval = max(1, int(cfg['seg'].get('seg_add_interval', 1)))
        frame_stride = max(mask_interval,
                           int((raw_stride + mask_interval - 1) / mask_interval) * mask_interval)
    else:
        frame_stride = 1
    processed_frame_ids = list(range(0, len(dataset), frame_stride))
    if processed_frame_ids[-1] != len(dataset) - 1:
        processed_frame_ids.append(len(dataset) - 1)
    print('[OnlineAnySeg] frame_stride=%d processed_frames=%d/%d' %
          (frame_stride, len(processed_frame_ids), len(dataset)), flush=True)

    scene_rep = Scene_rep(cfg, args, dataset, args.device)
    submap_manager = SubmapManager(
        cfg, dataset, args.device, pc_extractor=scene_rep.pc_extractor)
    last_valid_c2w = torch.eye(4)

    local_output_dir = str( os.path.join(args.output_dir, args.seq_name) )
    os.makedirs(local_output_dir, exist_ok=True)

    ########################################### main process ###########################################
    if frame_stride == 1:
        frame_stream = enumerate(dataloader)
    else:
        frame_stream = ((frame_id, dataset[frame_id]) for frame_id in processed_frame_ids)
    for frame_id, (color_img, depth_img, pose_c2w, seg_image, mask_features, seg_flag) in tqdm(frame_stream,
                                                                                              total=len(processed_frame_ids)):
        if max_frame_num is not None and frame_id >= max_frame_num:
            break

        # Direct indexed dataset access is unbatched; normalize it to the
        # existing batch-shaped tensors used by the mapping code.
        if color_img.ndim == 3:
            color_img = color_img.unsqueeze(0)
            depth_img = depth_img.unsqueeze(0)
            pose_c2w = pose_c2w.unsqueeze(0)
            seg_image = seg_image.unsqueeze(0)
            mask_features = mask_features.unsqueeze(0)
        if not torch.is_tensor(seg_flag):
            seg_flag = torch.tensor([bool(seg_flag)], device=color_img.device)
        elif seg_flag.ndim == 0:
            seg_flag = seg_flag.unsqueeze(0)

        # @color_img: Tensor(1, H , W, 3), dtype=float32, RGB
        # @depth_img: Tensor(1, H, W), dtype=float32
        # @pose_c2w: Tensor(1, 4, 4), dtype=float32
        # @seg_image: Tensor(1, H, W), dtype=uint8
        # @mask_features: Tensor(1, mask_num, 512), dtype=float32
        # @seg_flag: Tensor(1, ), dtype=bool, dtype=cpu

        # Step 1: preparation
        if torch.isnan(pose_c2w[0]).any().item() or torch.isinf(pose_c2w[0]).any().item():
            pose_c2w[0] = last_valid_c2w
        else:
            last_valid_c2w = pose_c2w[0]

        if seg_flag[0]:
            seg_image = seg_image[0]  # mask_id of each pixel, 0 means this pixel doesn't belongs to any mask, ndarray(H, W), dtype=uint8
            mask_features = mask_features[0]  # semantic feature extracted from 2D Foundation model

        # Step 2: do integration
        if frame_id % cfg["mapping"]["keyframe_freq"] == 0:
            with torch.no_grad():
                submap_manager.before_frame(
                    frame_id, depth_img[0], pose_c2w[0])
                # 2.1: fusing current frame
                pose_w2c = torch.inverse(pose_c2w[0])
                frustum_block_coords, extrinsic = scene_rep.integrate_frame(frame_id, color_img[0], depth_img[0], pose_w2c)  # integrate this frame to global PC

                if seg_flag[0] and frame_id % cfg["seg"]["seg_add_interval"] == 0:
                    # 2.2: collect masks as baseline instances or append-only
                    # observations owned by the submap manager.
                    if instance_owner == "submap":
                        valid_mask_ids, valid_mask_voxels = scene_rep.collect_seg_frame_evidence(
                            frame_id, depth_img[0], pose_c2w[0], frustum_block_coords,
                            seg_image, mask_features)
                    else:
                        valid_mask_ids, valid_mask_voxels = scene_rep.insert_seg_frame(
                            frame_id, color_img[0], depth_img[0], pose_c2w[0],
                            frustum_block_coords, seg_image, mask_features)
                    submap_manager.observe(
                        frame_id, color_img[0], depth_img[0], pose_c2w[0],
                        seg_image, mask_features,
                        merged_pairs=scene_rep.last_inserted_mask_pairs,
                        scene_feature_points=scene_rep.pc_xyz,
                        scene_feature_values=scene_rep.pc_feature)

                    if instance_owner == "baseline":
                        submap_manager.graph_update(
                            frame_id, scene_rep.maskGraph.instance_dict)
                        # 2.3: periodically merge baseline global masks.
                        if frame_id > 0 and frame_id % scene_rep.merge_frame_interval == 0:
                            scene_rep.update_masks(frame_id)
                            submap_manager.graph_update(
                                frame_id, scene_rep.maskGraph.instance_dict)

        # Step 2.4: update visualized segmentation result
        if scene_rep.vis_pc_flag:
            scene_rep.vis_pc.update()

        # Step 2.5: save latest ckpt
        if (instance_owner == "baseline" and cfg["save"]["ckpt_interval"] > 0
                and frame_id > 0 and frame_id % cfg["save"]["ckpt_interval"] == 0):
            ckpt_save_dir = os.path.join(local_output_dir, "ckpt_%d" % frame_id)
            os.makedirs(ckpt_save_dir, exist_ok=True)
            scene_rep.save_ckpt(frame_id, ckpt_save_dir)


    # Step 3: save final segmentation results
    print("################ Begin to save final ckpt and seg mesh/pointcloud...")
    frame_id_final = (processed_frame_ids[-1] if max_frame_num is None
                      else min(processed_frame_ids[-1], max_frame_num - 1))

    # Baseline owns the final global merge.  In submap-owner mode there is no
    # global MaskGraph merge to flush; SubmapManager is the sole instance owner.
    if (instance_owner == "baseline" and frame_id_final > 0 and
            frame_id_final % scene_rep.merge_frame_interval != 0):
        scene_rep.update_masks(frame_id_final)
        submap_manager.graph_update(
            frame_id_final, scene_rep.maskGraph.instance_dict)

    ckpt_save_dir = os.path.join(local_output_dir, "ckpt_%d" % frame_id_final)
    os.makedirs(ckpt_save_dir, exist_ok=True)

    # Save geometry and choose the sole instance owner.
    final_ckpt_path = os.path.join(local_output_dir, "ckpt_scene_rep_final.npz")
    if instance_owner == "baseline":
        scene_rep.save_ckpt(frame_id_final, ckpt_save_dir, filter_flag=True, ckpt_path=final_ckpt_path)
        if not os.path.exists(final_ckpt_path):
            point_count = int(scene_rep.points.shape[0])
            feature_dim = int(cfg["mask"]["feature_dim"])
            np.savez(final_ckpt_path, pred_masks=np.zeros((point_count, 0), dtype=bool),
                     pred_score=np.ones((0,), dtype=np.float64),
                     pred_classes=np.zeros((0,), dtype=np.int32),
                     pred_sem_features=np.zeros((feature_dim, 0), dtype=np.float32))
    else:
        # Geometry-only checkpoint: instances are exported by SubmapManager.
        scene_rep.points, scene_rep.colors = scene_rep.get_pc()
        scene_rep.pc_frame_id = frame_id_final
        point_count = int(scene_rep.points.shape[0])
        feature_dim = int(cfg["mask"]["feature_dim"])
        np.savez(final_ckpt_path, pred_masks=np.zeros((point_count, 0), dtype=bool),
                 pred_score=np.ones((0,), dtype=np.float64),
                 pred_classes=np.zeros((0,), dtype=np.int32),
                 pred_sem_features=np.zeros((feature_dim, 0), dtype=np.float32))
        print("Saved geometry-only checkpoint:", final_ckpt_path)

    if instance_owner == "baseline":
        scene_rep.save_merging_result(frame_id_final, reextract=False)

    # save finally reconstructed pointcloud
    final_pc_path = os.path.join(local_output_dir, "final.ply")
    scene_rep.save_pc(scene_rep.points, scene_rep.colors, final_pc_path)

    scene_rep.finalize_fcgf_diagnostics()
    submap_manager.save_outputs(
        local_output_dir,
        global_points=scene_rep.points,
        global_colors=scene_rep.colors)
    if not submap_manager.enabled:
        # Keep the standard output contract for configurations that do not
        # enable object submaps.
        shutil.copyfile(
            os.path.join(local_output_dir, "ckpt_scene_rep_final.npz"),
            os.path.join(local_output_dir, "ckpt_final.npz"))

    print("Input sequence finished !")
