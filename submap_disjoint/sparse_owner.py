"""Sparse candidate implementation of the exact disjoint submap owner.

Keep the reference owner, features, voxel history, lifetime and export intact.
Only containment storage, candidate algebra and remapping differ.
"""
from submap_disjoint.owner import *
from submap_disjoint.sparse_edges import *

class SparseDisjointSubmapManager(DisjointSubmapManager):
    def __init__(self, cfg, args, dataset, geometry, device="cuda:0"):
        self.edges = Edges()
        self.sparse_audit = dict(direct_candidates=0, supporter_candidates=0,
            supporter_only_candidates=0, possible_dense_pairs=0,
            dense_workspace_calls=0, audit_dense_calls=0,
            peak_edges=0, edge_bytes_peak_estimate=0,
            candidate_seconds=0., observe_seconds=0., observations=0,
            verify_calls=0, verify_max_score_error=0.)
        super().__init__(cfg,args,dataset,geometry,device)
        self.contain_matrix = None
        self.maskGraph.contain_matrix = None
        self.verify_sparse = bool(cfg.get("sparse_candidate",{}).get("verify",False))
        self.merge_selection = cfg.get('ig_submap',{}).get('merge_selection','disjoint_pairs')
        if self.merge_selection not in ('disjoint_pairs','baseline_components'):
            raise ValueError('Unsupported IG-Submap merge selection: '+str(self.merge_selection))

    @property
    def get_containing_mat(self):
        return self.edges

    @property
    def edge_values(self):
        return torch.tensor(list(self.edges.values.values()),dtype=torch.float32)

    def pack_edges(self):
        if len(self.edges.values)>self.sparse_audit['peak_edges']:
            self.sparse_audit['peak_edges']=len(self.edges.values)
            self.sparse_audit['edge_bytes_peak_estimate']=self.edges.nbytes_estimate()

    def unpack_edges(self):
        assert self.contain_matrix is None

    def verify_selection(self, pairs, valid_ids=None):
        # Optional diagnostic ONLY: compare sparse selectors with the frozen
        # GPU dense equations on the very same evidence state.
        c=self.edges.dense_for_audit(self.c_mask_num,self.device)
        self.sparse_audit['audit_dense_calls']+=1
        if valid_ids is None:
            f,*_=self.metrics.compute_final_sim_mat(c,self.get_mask_features,
                self.get_mask_geo_features,use_IoU=True)
            tri=create_lower_triangular_matrix(self.c_mask_num,device=self.device)
            sim=tri&(f>self.sim_merge_thresh)&(c>self.merge_overlap_thresh)&(c.T>self.merge_overlap_thresh)
            a=set_diagonal_to_zero(((c.T>self.containing_ratio)&(c>self.contained_ratio)).float())
            support=a@torch.diag(self.get_merged_mask_weight)@a.T
            selected=sim|((support*tri.float())>=self.merge_supporter_num)
        else:
            c=mask_matrix_rows_and_cols(c,valid_ids)
            cm=(c>self.merge_contain_ratio).to(c)*c
            cr=retain_max_per_column(set_diagonal_to_zero(cm))
            selected=cr.bool()
            if self.with_feature:
                f,*_=self.metrics.compute_final_sim_mat(c,self.get_mask_features,
                    self.get_mask_geo_features,use_IoU=False)
                f=mask_matrix_rows_and_cols(f,valid_ids)
                for ratio,threshold in zip(self.merge_contain_ratio_feat_list,self.contain_feature_sim_thresh_list):
                    selected |= (cr>ratio)&(f>threshold)
        selected=set_row_and_column_zero(selected,self.get_mask_wo_geo_feature,self.get_mask_wo_geo_feature)
        expected={tuple(p) for p in selected.nonzero().cpu().tolist()}
        assert set(pairs)==expected, ('sparse/dense candidate mismatch',set(pairs)^expected)
        self.sparse_audit['verify_calls']+=1

    def update_masks(self, frame_id):
        self.update_mask_geo_features(frame_id)
        started=time.perf_counter()
        pairs=select_initial(self)
        self.sparse_audit['candidate_seconds']+=time.perf_counter()-started
        if self.verify_sparse: self.verify_selection(pairs)
        valid,_=self.merge_masks(frame_id,pairs,self.edges,count_merge=True)
        if self.save_interval>0 and frame_id%self.save_interval==0:
            self.save_merging_result(frame_id,valid)
        if valid is not None and valid.numel():
            for iteration in range(self.merge_contain_iter_num):
                pairs=select_containment(self,valid)
                if self.verify_sparse: self.verify_selection(pairs,valid)
                valid,changed=self.merge_masks(frame_id,pairs,self.edges,count_merge=False)
                if not changed: break
        self.select_masks_to_remove(frame_id)

    def observe(self, *args, **kwargs):
        started=time.perf_counter()
        super().observe(*args,**kwargs)
        self.sparse_audit['observe_seconds']+=time.perf_counter()-started
        self.sparse_audit['observations']+=1

    def save_outputs(self, frame_id, directory):
        super().save_outputs(frame_id,directory)
        from pathlib import Path
        (Path(directory)/'sparse_candidate_metrics.json').write_text(
            json.dumps(self.sparse_audit,indent=2))

    def insert_seg_frame(self, frame_id, color_img, depth_img, pose_c2w, frustum_block_coords_o3c, seg_image, mask_sem_features, mask_min_size=50):
        self.last_inserted_mask_pairs = []
        # Step 1: get corresponding blocks and their containing voxels in VoxelBlockGrid of current frame
        cur_block_indices_o3c, _ = self.voxel_block_grids.hashmap().find(frustum_block_coords_o3c)  # Tensor(N, )
        o3c.cuda.synchronize()
        voxel_coords_o3c, voxel_indices_o3c = self.voxel_block_grids.voxel_coordinates_and_flattened_indices(cur_block_indices_o3c)  # voxel coordinates and 1D indices contained by given blocks

        frustum_block_coords = torch.utils.dlpack.from_dlpack(frustum_block_coords_o3c.to_dlpack())
        cur_block_indices = torch.utils.dlpack.from_dlpack(cur_block_indices_o3c.to_dlpack())  # *** key in hashmap of each selected VoxelBlock, Tensor(s_block_num, ), dtype=int32
        voxel_coords = torch.utils.dlpack.from_dlpack(voxel_coords_o3c.to_dlpack())  # *** voxel coordinates in selected blocks(in World CS), Tensor(c_voxel_num, 3), dtype=float32
        voxel_indices = torch.utils.dlpack.from_dlpack(voxel_indices_o3c.to_dlpack())  # 1D indices of each voxel coordinate (in World CS)

        self.set_active_frustum(frame_id, voxel_coords, voxel_indices)  # update active frame frustum

        # Step 2: find valid 2D masks in this frame, and compute their corresponding voxels in 3D (coordinates in World CS) by back-projecting
        # 2.1: back-project each 2D mask in this frame to 3D
        mask_dict, valid_mask_ids, _, valid_mask_sem_features = self.voxel_grids.turn_mask_to_voxel(frame_id, depth_img, pose_c2w, voxel_coords, seg_image, mask_sem_features)
        self.last_visibility_voxels = dict(mask_dict)
        self.last_visibility_voxel_indices = {int(mid): self.voxel_grids.voxel_coords2voxel_indices(coords) for mid, coords in mask_dict.items()}
        mask_dict, mask_indices_dict, mask_in_frame_list = self.maskGraph.add_frame_masks(frame_id, mask_dict)  # compute each voxel's corresponding mask_ID in this frame

        # 2.2: check whether each 3D mask is valid
        valid_mask_ids_new = []
        mask_dict_new = {}
        valid_mask_sem_features_new = []
        mask_indices_dict_new = {}
        mask_in_frame_list_new = []
        for mask_id, mask_sem_feature, mask_in_frame in zip(valid_mask_ids, valid_mask_sem_features, mask_in_frame_list):
            mask_coords = mask_dict[mask_id]
            mask_indices = mask_indices_dict[mask_id]
            mask_size = mask_coords.shape[0]
            if mask_size > mask_min_size:
                valid_mask_ids_new.append(mask_id)
                mask_dict_new[mask_id] = mask_coords
                valid_mask_sem_features_new.append(mask_sem_feature)
                mask_indices_dict_new[mask_id] = mask_indices
                mask_in_frame_list_new.append(mask_in_frame)

        valid_mask_ids = valid_mask_ids_new
        mask_dict = mask_dict_new
        valid_mask_sem_features = valid_mask_sem_features_new
        mask_indices_dict = mask_indices_dict_new
        mask_in_frame_list = mask_in_frame_list_new
        self.last_mask_voxel_indices = dict(mask_indices_dict)

        mask_num_this_frame = len(valid_mask_ids)
        if mask_num_this_frame == 0:
            return [], []

        # Step 3: for each newly detected 2D mask, fill the new rows and new cols in Containing_matrix
        for i in range(mask_num_this_frame):
            mask_id = valid_mask_ids[i]
            f_mask_id = (frame_id, mask_id)
            merged_mask_id = self.c_mask_num + i  # 1D index in all merged masks
            mask_sem_feature = valid_mask_sem_features[i]  # visual embedding of this mask, Tensor(feature_dim, )

            # 3.1: fill related vars and create a new Instance obj for this new mask
            mask_voxel_coords = mask_dict[mask_id]
            mask_voxel_indices = mask_indices_dict[mask_id]

            self.maskGraph.insert_instance(frame_id, mask_id, merged_mask_id, mask_voxel_coords, mask_voxel_indices, mask_sem_feature)
            self.maskGraph.f_mask_voxels_clouds[f_mask_id] = mask_voxel_coords
            self.maskGraph.merged_mask_in_frame_mat[merged_mask_id] = mask_in_frame_list[i]
            self.mask_voxels_coords[merged_mask_id] = mask_voxel_coords
            self.mask_features[merged_mask_id] = mask_sem_feature

            # 3.2: compute contain_ratio between this new mask and existing masks
            if len(self.global_f_mask_ids) == 0:
                # for 1st seg image
                self.edges.set(i, i, 1.)
            else:
                # for >=2nd seg image,
                # 3.2.1: for this mask, compute other overlapping masks and their overlap voxel num
                overlap_mask_ids, contain_ratio_q2o, contain_ratio_o2q = self.maskGraph.query_mask_under_visible_part(mask_in_frame_list[i], mask_voxel_coords, self.global2merged_mask_id)

                if overlap_mask_ids.numel():
                    self.edges.write_row_col(merged_mask_id, overlap_mask_ids,
                                             contain_ratio_q2o, contain_ratio_o2q)
                self.edges.set(merged_mask_id, merged_mask_id, 1.)
        # END for

        # Step 4: for each valid 2D mask in current frame, update corresponding vars
        frame_valid_mask_ids = []  # merged_mask_id of each newly added mask in this frame
        frame_valid_mask_voxels = []
        frame_global_mask_id = []
        for i in range(mask_num_this_frame):
            mask_id = valid_mask_ids[i]  # mask_ID of this 2D mask in current frame, int
            f_mask_id = (frame_id, mask_id)
            mask_voxels = mask_dict[mask_id]  # Voxel coords of this 2D mask, Tensor(n_i, 3)
            merged_mask_id = self.c_mask_num  # merged (1D) mask_ID of this original 2D mask in merged mask list
            global_mask_id = len(self.global_f_mask_ids)
            frame_global_mask_id.append(global_mask_id)

            frame_valid_mask_ids.append(merged_mask_id)
            frame_valid_mask_voxels.append(mask_voxels)

            # 4.1: update its global_mask_id and merged_mask_id
            self.global_f_mask_ids.append(f_mask_id)
            self.global2merged_mask_id.append(merged_mask_id)  # for new 2D masks in this frame, its merged_mask_ID is the current largest merged_mask_ID + 1
            self.last_inserted_mask_pairs.append({
                "mask_id": int(mask_id),
                "global_mask_id": int(global_mask_id),
                "merged_mask_id": int(merged_mask_id),
            })
            self.merged_mask_size.append(mask_voxels.shape[0])
            self.merged_mask_weight[merged_mask_id] = 1.
            self.merged_mask_last_frame[merged_mask_id] = frame_id

            # 4.2: update voxel hashing table
            self.voxel_hash_table.insert_mask_voxels(mask_voxels, global_mask_id)  # ***
            self.c_mask_num += 1
            self.g_mask_num += 1

        # Step 5: for each newly added mask, extract its geometric feature by querying
        if self.get_mask_w_geo_feature.shape[0] == 0:
            mask_voxels_list = list(self.mask_voxels_coords.values())
            mask_id_list = [i for i in range(len(mask_voxels_list))]
        else:
            mask_voxels_list = frame_valid_mask_voxels
            mask_id_list = frame_valid_mask_ids

        extract_flag, mask_geo_feature_list, pts_feature, pts_xyz = self.pc_extractor.get_masks_geometric_features(mask_voxels_list, scene_points=self.get_pc_xyz)
        self._record_fcgf_diagnostics(frame_id, mask_id_list, "insert")
        if extract_flag:
            for i, mask_geo_feature in enumerate(mask_geo_feature_list):
                if mask_geo_feature is None:
                    continue
                merged_mask_id = mask_id_list[i]
                self.mask_geo_features[merged_mask_id] = mask_geo_feature
                self.mask_geo_features_mask[merged_mask_id] = 1.

        # save extracted feature pointcloud
        if extract_flag and pts_feature is not None and pts_xyz is not None:
            self.pc_xyz = pts_xyz
            self.pc_feature = pts_feature
            self.pc_feature_frame_id = frame_id

        # Step 6: visualize current color image
        if self.vis_color_flag:
            self.vis_c.update(color_img)

        return frame_valid_mask_ids, frame_valid_mask_voxels

    def merge_masks(self, frame_id, merge_mat, c_mat, count_merge=False, full_geo_merge=True):
        if not merge_mat:
            return None, False
        c_mask_num_before = self.c_mask_num
        merged_mask_id_clusters = sparse_clusters(
            merge_mat, self.c_mask_num, c_mat, self.migration_audit, mode=self.merge_selection)

        # (1) assign a new merged_mask_id for each cluster, and reassign other merged masks' merged_mask_id;
        # (2) in containing matrix, firstly remove rows (and cols) corresponding to masks that need to be merged, then append new rows (and cols) for merged masks to the end;
        # (3) update members: global2merged_mask_id, merged_mask_size, mask_voxels_indices, merged_mask_weight

        # Step 1: identify rows(and cols) to keep and to merge
        rows_to_keep = []  # list of int
        rows_to_merge = []  # list of list
        for merged_mask_id_cluster in merged_mask_id_clusters:
            if merged_mask_id_cluster.shape[0] == 1:
                rows_to_keep.append(merged_mask_id_cluster.item())
            else:
                rows_to_merge.append(merged_mask_id_cluster)

        kept_mask_num, merged_mask_num = len(rows_to_keep), len(rows_to_merge)
        merged_mask_num_new = kept_mask_num + merged_mask_num  # total mask number after this merging
        rc_to_keep = torch.tensor(rows_to_keep, dtype=torch.long)

        # case: No masks are merged in this iteration
        if merged_mask_num == 0:
            return None, False

        # Step 2: reassign merged_mask_id for each kept mask and merged mask
        self.maskGraph.keep_and_merge_masks(frame_id, c_mask_num_before, rows_to_keep, rows_to_merge)

        # 2.1: preparation, create new tensors
        global2merged_mask_id = torch.tensor(self.global2merged_mask_id, dtype=torch.long)  # Tensor(collected_global_mask_num, )
        global2merged_mask_id_new = -1 * torch.ones_like(global2merged_mask_id)
        c_mat_new = Edges({(i,i):1. for i in range(merged_mask_num_new)})  # Tensor(m_mask_num, m_mask_num)
        mask_voxels_coords_new = {}
        mask_features_new = torch.zeros((c_mask_num_before, self.cfg["mask"]["feature_dim"]), dtype=torch.float32, device=self.device)
        mask_geo_features_new = torch.zeros((c_mask_num_before, self.cfg["pc_extractor"]["feature_dim"]), dtype=torch.float32, device=self.device)
        mask_geo_features_mask_new = torch.zeros((c_mask_num_before, ), dtype=torch.float32, device=self.device)  # whether each merged mask's geometric feature has been extracted successfully
        merged_mask_size_new = []
        merged_mask_weight_new = torch.zeros((self.t_mask_num,), dtype=torch.float32, device=self.device)
        merged_mask_last_frame_new = -1 * torch.ones((self.t_mask_num,), dtype=torch.int64, device=self.device)

        # 2.2: re-assign merged_mask_id to each kept mask, and copy their corresponding rows & cols from current c_mat to new c_mat
        c_mat_to_keep = c_mat.subset(rc_to_keep)
        c_mat_new.values.update(c_mat_to_keep.values)

        new_mask_counter = 0
        for mask_id_kept in rows_to_keep:
            corr_global_mask_ids = torch.where(global2merged_mask_id == mask_id_kept)[0]
            global2merged_mask_id_new[corr_global_mask_ids] = new_mask_counter

            mask_voxels_coords_new[new_mask_counter] = self.mask_voxels_coords[mask_id_kept]
            mask_features_new[new_mask_counter] = self.get_mask_features[mask_id_kept]
            mask_geo_features_new[new_mask_counter] = self.get_mask_geo_features[mask_id_kept]
            mask_geo_features_mask_new[new_mask_counter] = self.get_mask_geo_features_mask[mask_id_kept]
            merged_mask_size_new.append(self.merged_mask_size[mask_id_kept])
            merged_mask_weight_new[new_mask_counter] = self.get_merged_mask_weight[mask_id_kept]
            merged_mask_last_frame_new[new_mask_counter] = self.get_merged_mask_last_frame[mask_id_kept]
            new_mask_counter += 1

        # 2.3: for each new merged mask, judge whether its is valid, and reassign a new merged_mask_id, and fill new c_mat in region (2) and (3)
        global2merged_mask_id_new_list = global2merged_mask_id_new.tolist()
        valid_merge_mask_ids = global2merged_mask_id_new[global2merged_mask_id_new != -1]
        valid_merge_mask_ids = torch.unique(valid_merge_mask_ids).tolist()  # mask_IDs of all kept merged masks

        merged_mask_voxel_coords_list = []  # voxel coordinates of all valid merged mask
        merged_mask_ids_new = []
        merged_mask_num_valid = 0
        rows_to_merge_valid = []
        mask_cluster_list = []
        for mask_cluster in rows_to_merge:
            merged_mask_id_new = new_mask_counter

            # 2.3.1: compute instance mask of this merged mask
            mask_cluster = mask_cluster.tolist()
            mask_cluster_list.append(mask_cluster)
            cluster_voxel_list = [self.mask_voxels_coords[mask_id] for mask_id in mask_cluster]
            cluster_voxel_coords = torch.concat(cluster_voxel_list, dim=0)
            cluster_voxel_coords = torch.unique(cluster_voxel_coords, dim=0)  # Voxel coordinates of this merged mask, Tensor(new_mask_size, 3)

            # 2.3.2: compute merged semantic feature of this merged mask
            cluster_features = self.get_mask_features[mask_cluster]
            cluster_feature_merged = cluster_features.sum(dim=0)

            # 2.3.3: update self.mask_voxels_indices, self.merged_mask_weight
            merged_mask_num_valid += 1
            merged_mask_ids_new.append(merged_mask_id_new)
            merged_mask_voxel_coords_list.append(cluster_voxel_coords)
            rows_to_merge_valid.append(mask_cluster)
            mask_voxels_coords_new[merged_mask_id_new] = cluster_voxel_coords
            mask_features_new[merged_mask_id_new] = cluster_feature_merged

            merged_mask_weight_new[merged_mask_id_new] = torch.sum(self.get_merged_mask_weight[mask_cluster])  # *** reassign weight of this merged mask
            merged_mask_last_frame_new[merged_mask_id_new] = frame_id

            # 2.3.4: add new rows and cols for this merged mask to all kept merged masks (region (2) and (3))
            # ***Attention: 这里先只考虑计算该merged mask与其他kept masks之间的containing ratio
            merged_mask_frame = self.maskGraph.merged_mask_in_frame_mat[merged_mask_id_new]
            overlap_mask_ids, containing_ratio, contained_ratio = self.maskGraph.query_mask_under_visible_part(merged_mask_frame, cluster_voxel_coords, global2merged_mask_id_new_list, valid_merge_mask_ids)

            if overlap_mask_ids.numel():
                c_mat_new.write_row_col(merged_mask_id_new, overlap_mask_ids,
                                        containing_ratio, contained_ratio)

            new_mask_counter += 1

        # 2.4: for each new merged mask, re-compute its geometric feature, fill new c_mat in region (4), and update self.global2merged_mask_id
        if full_geo_merge and frame_id == self.pc_feature_frame_id:
            _, merged_mask_geo_feature_list, _, _ = self.pc_extractor.get_masks_geometric_features(merged_mask_voxel_coords_list, scene_points=self.pc_xyz, pts_feature=self.pc_feature)
            self._record_fcgf_diagnostics(frame_id, merged_mask_ids_new, "merge")
        else:
            merged_mask_geo_feature_list = [None for _ in range(merged_mask_num_valid)]

        for i in range(merged_mask_num_valid):
            merged_mask_id_i = merged_mask_ids_new[i]
            mask_voxel_coords_i = merged_mask_voxel_coords_list[i]  # Tensor(voxel_num_i, 3)

            # 2.4.1: self.merged_mask_size
            mask_cluster_tensor = torch.tensor(rows_to_merge_valid[i])
            corr_global_mask_ids = torch.where(torch.isin(global2merged_mask_id, mask_cluster_tensor))[0]
            global2merged_mask_id_new[corr_global_mask_ids] = merged_mask_id_i  # update global2merged_mask_id
            merged_mask_size_new.append(mask_voxel_coords_i.shape[0])

            # 2.4.2: re-compute geometric feature for this newly merged mask, and update
            merged_mask_geo_feature = merged_mask_geo_feature_list[i]
            merged_mask_geo_feature_mask = 1.
            if merged_mask_geo_feature is None:
                merged_mask_geo_feature, merged_mask_geo_feature_mask = self.aggregate_masks_geo_features(mask_cluster_list[i])
            mask_geo_features_new[merged_mask_id_i] = merged_mask_geo_feature
            mask_geo_features_mask_new[merged_mask_id_i] = merged_mask_geo_feature_mask

        # 2.4.3: re-compute containing ratio between merged masks, i.e. fill new c_mat in region (4)
        c_mat_new_instances = compute_sparse_containment(
            self.maskGraph, merged_mask_ids_new, global2merged_mask_id_new.tolist())
        # Replace the block, including zero diagonal entries, exactly.
        block_ids = set(merged_mask_ids_new)
        c_mat_new.values = {k:v for k,v in c_mat_new.values.items()
                            if not (k[0] in block_ids and k[1] in block_ids)}
        c_mat_new.insert_block(c_mat_new_instances, merged_mask_ids_new)

        # 2.5: set new tensors as members
        merged_mask_num_new_valid = kept_mask_num + merged_mask_num_valid
        self.global2merged_mask_id = global2merged_mask_id_new.tolist()
        self.edges = c_mat_new
        self.c_mask_num = merged_mask_num_new_valid
        self.mask_voxels_coords = mask_voxels_coords_new
        self.mask_features[:c_mask_num_before] = mask_features_new
        self.mask_geo_features[:c_mask_num_before] = mask_geo_features_new
        self.mask_geo_features_mask[:c_mask_num_before] = mask_geo_features_mask_new
        self.merged_mask_size = merged_mask_size_new
        self.merged_mask_weight = merged_mask_weight_new
        self.merged_mask_last_frame = merged_mask_last_frame_new

        # Step 3: select valid masks by weight
        mask_weight_threshold = min(self.merge_time + 2, self.cfg["seg"]["mask_weight_threshold"])
        valid_merged_mask_ids = torch.where(self.merged_mask_weight[:self.c_mask_num] >= mask_weight_threshold)[0]

        if count_merge:
            self.merge_time += 1

        return valid_merged_mask_ids, True

    def remove_masks(self, frame_id, remove_mask_ids):
        # Step 1: find mask_IDs of kept masks
        all_mask_ids = torch.arange(0, self.c_mask_num, dtype=remove_mask_ids.dtype).to(self.device)
        kept_mask_ids = compute_complementary(all_mask_ids, remove_mask_ids)
        kept_mask_num = kept_mask_ids.shape[0]

        # Step 2: update mask info in self.maskGraph obj
        kept_mask_ids_list = [kept_mask_id.item() for kept_mask_id in kept_mask_ids]
        mask_num_new = self.maskGraph.keep_and_merge_masks(frame_id, self.c_mask_num, kept_mask_ids_list)

        # Step 3: update corresponding members
        # 3.1: preparation, create new tensors
        c_mat = self.get_containing_mat
        c_mask_num_before = self.c_mask_num
        global2merged_mask_id = torch.tensor(self.global2merged_mask_id, dtype=torch.long)  # Tensor(collected_global_mask_num, )
        global2merged_mask_id_new = -1 * torch.ones_like(global2merged_mask_id)
        mask_voxels_coords_new = {}
        mask_features_new = torch.zeros((c_mask_num_before, self.cfg["mask"]["feature_dim"]), dtype=torch.float32, device=self.device)
        mask_geo_features_new = torch.zeros((c_mask_num_before, self.cfg["pc_extractor"]["feature_dim"]), dtype=torch.float32, device=self.device)
        mask_geo_features_mask_new = torch.zeros((c_mask_num_before,), dtype=torch.float32, device=self.device)  # whether each merged mask's geometric feature has been extracted successfully
        merged_mask_size_new = []
        merged_mask_weight_new = torch.zeros((self.t_mask_num,), dtype=torch.float32, device=self.device)
        merged_mask_last_frame_new = -1 * torch.ones((self.t_mask_num,), dtype=torch.int64, device=self.device)

        # 3.2: for each mask needs to be removed, find their raw (/global) masks, and set their mapping mask_IDs to -1
        r_global_mask_ids = torch.where( torch.isin(global2merged_mask_id, remove_mask_ids.to(global2merged_mask_id)) )[0]
        global2merged_mask_id_new[r_global_mask_ids] = -1

        # 3.3: re-assign merged_mask_id to each kept mask, and copy their corresponding rows & cols from current c_mat to new c_mat
        c_mat_new = c_mat.subset(kept_mask_ids)

        new_mask_counter = 0
        kept_mask_ids = kept_mask_ids.tolist()
        for i, mask_id_kept in enumerate(kept_mask_ids):
            corr_global_mask_ids = torch.where(global2merged_mask_id == mask_id_kept)[0]
            global2merged_mask_id_new[corr_global_mask_ids] = new_mask_counter

            mask_voxels_coords_new[new_mask_counter] = self.mask_voxels_coords[mask_id_kept]
            mask_features_new[new_mask_counter] = self.get_mask_features[mask_id_kept]
            mask_geo_features_new[new_mask_counter] = self.get_mask_geo_features[mask_id_kept]
            mask_geo_features_mask_new[new_mask_counter] = self.get_mask_geo_features_mask[mask_id_kept]
            merged_mask_size_new.append(self.merged_mask_size[mask_id_kept])
            merged_mask_weight_new[new_mask_counter] = self.get_merged_mask_weight[mask_id_kept]  # set as former weight value rather than 1
            merged_mask_last_frame_new[new_mask_counter] = self.get_merged_mask_last_frame[mask_id_kept]
            new_mask_counter += 1

        # 3.4: set new tensors as members
        self.global2merged_mask_id = global2merged_mask_id_new.tolist()
        self.edges = c_mat_new
        self.c_mask_num = kept_mask_num
        self.mask_voxels_coords = mask_voxels_coords_new
        self.mask_features[:c_mask_num_before] = mask_features_new
        self.mask_geo_features[:c_mask_num_before] = mask_geo_features_new
        self.mask_geo_features_mask[:c_mask_num_before] = mask_geo_features_mask_new
        self.merged_mask_size = merged_mask_size_new
        self.merged_mask_weight = merged_mask_weight_new
        self.merged_mask_last_frame = merged_mask_last_frame_new
