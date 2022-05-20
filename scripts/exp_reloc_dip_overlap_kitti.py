#!/usr/bin/env python3
# Developed by Yiming Wang and Fabio Poiesi
# Covered by the LICENSE file in the root of this project.
# Brief: This script generates the transform between the candidate loop frames and each reference frame based on
# the pre-computed DIP local 3D descriptors.
# Experiment setup: this experiment follows the relocalisation setup
# where camera poses are not used to define the candidate frame.
# At each frame, all the previously seen frames can be candidates.
# To constrain the computation load, every 20 frame will be considered as a candidate.
# The 'overlapnet_gt_est_reloc_result_20.json' includes all the candidate pairs obtained
# using the above-mentioned protocol, with the Overlapnet result included.

import open3d as o3d
import numpy as np
import os
import time
import json
from dip_transform_overlap import com_overlap_transform, com_overlap_mnn


# this function computes RON with the estimated transform between candidate and ref
def compute_mnn(root_dir_kitti, dip_folder, json_file, pair_gap):
    if os.path.exists(json_file):
        json_data = json.load(open(json_file))
    else:
        print("No result json file, use input json data to create.")

    for i, key in enumerate(json_data):
        data = json_data[key]
        if 'dip_mnn' in data:
            print("{} is generated, skip...".format(key))
            continue
        else:
            print("Generate for {} ...".format(key))
            json_data[key]["dip_mnn"] = []
            json_data[key]["dip_ron"] = []

            transformations = data["transformations"]
            pairs = data['indices_pairs']
            if len(pairs) > 0:
                print("Pairs for {} is not none, compute for each pair".format(key))
                for k, pair in enumerate(pairs):
                    if k % pair_gap == 0:
                        print("performing each valid pair that is dividable by {} at pair index {}".format(str(pair_gap), str(k)))

                        pcd_current_path = os.path.join(root_dir_kitti, '{:06d}.bin'.format(pair[1]))
                        pcd_ref_path = os.path.join(root_dir_kitti, '{:06d}.bin'.format(pair[0]))

                        dip_current_path = os.path.join(dip_folder, '{:06d}.npz'.format(pair[1]))
                        dip_ref_path = os.path.join(dip_folder, '{:06d}.npz'.format(pair[0]))

                        # load and prepare for the current pcd
                        xyzr_current = np.fromfile(pcd_current_path, dtype=np.float32).reshape(-1, 4)
                        xyz_current = xyzr_current[:, :3]
                        pcd_current = o3d.geometry.PointCloud()
                        pcd_current.points = o3d.utility.Vector3dVector(xyz_current)

                        data_dip_current = np.load(dip_current_path)  # current frame
                        inds_current = data_dip_current['inds']
                        pcd_current_desc = data_dip_current['pcd1_desc']
                        # subsample the pcd
                        sampling_ratio = 0.5
                        random_selected_indices = np.random.choice(pcd_current_desc.shape[0], int(sampling_ratio*pcd_current_desc.shape[0]))
                        pcd_current_desc = pcd_current_desc[random_selected_indices, :]
                        inds_current = inds_current[random_selected_indices]
                        pcd_current_pts = np.asarray(pcd_current.points)[inds_current]


                        # fix the NAN issue
                        bool_nan_current = np.isnan(pcd_current_desc.sum(1))
                        pcd_current_desc = pcd_current_desc[~bool_nan_current, :]
                        pcd_current_pts = pcd_current_pts[~bool_nan_current]

                        _pcd_current = o3d.geometry.PointCloud()
                        _pcd_current.points = o3d.utility.Vector3dVector(pcd_current_pts)

                        # load and prepare for the ref pcd
                        xyzr_ref = np.fromfile(pcd_ref_path, dtype=np.float32).reshape(-1, 4)
                        xyz_ref = xyzr_ref[:, :3]
                        pcd_ref = o3d.geometry.PointCloud()
                        pcd_ref.points = o3d.utility.Vector3dVector(xyz_ref)

                        data_dip_ref = np.load(dip_ref_path)  # current frame
                        inds_ref = data_dip_ref['inds']
                        pcd_ref_desc = data_dip_ref['pcd1_desc']

                        # subsample the pcd
                        random_selected_indices = np.random.choice(pcd_ref_desc.shape[0], int(sampling_ratio*pcd_ref_desc.shape[0]))
                        pcd_ref_desc = pcd_ref_desc[random_selected_indices, :]
                        inds_ref = inds_ref[random_selected_indices]
                        pcd_ref_pts = np.asarray(pcd_ref.points)[inds_ref]

                        # fix the NAN issue
                        bool_nan_ref = np.isnan(pcd_ref_desc.sum(1))
                        pcd_ref_desc = pcd_ref_desc[~bool_nan_ref, :]
                        pcd_ref_pts = pcd_ref_pts[~bool_nan_ref]

                        _pcd_ref = o3d.geometry.PointCloud()
                        _pcd_ref.points = o3d.utility.Vector3dVector(pcd_ref_pts)

                        pose_c2r = np.asarray(transformations[k])
                        # function to compute MNN
                        ratio_mnn, ron = com_overlap_mnn(pcd_current_desc, pcd_ref_desc, _pcd_current, _pcd_ref, pose_c2r, 1)
                        print("MNN ratio {}, RON {}, overlap {}".format(str(ratio_mnn), str(ron), str(json_data[key]["dip_overlap"][k])))
                        json_data[key]["dip_mnn"].append(ratio_mnn)
                        json_data[key]["dip_ron"].append(ron)
                    else:
                        json_data[key]["dip_mnn"].append(0)
                        json_data[key]["dip_ron"].append(0)

                # update the json file for each frame
                print("Write for {} in the json file, only write when pair larger than 0".format(key))
                with open(json_file, 'w') as fp:
                    json.dump(json_data, fp)
    return json_data


# this function computes the transform and its fitness with RANSAC using the preprocessed DIP descriptors
def compute_transformation_once(root_dir_kitti, dip_folder, current, ref):
    # load lidar data
    xyzr_current = np.fromfile(os.path.join(root_dir_kitti, '{:06d}.bin'.format(current)), dtype=np.float32).reshape(-1, 4)
    xyz_current = xyzr_current[:, :3]
    pcd_current = o3d.geometry.PointCloud()
    pcd_current.points = o3d.utility.Vector3dVector(xyz_current)

    xyzr_ref = np.fromfile(os.path.join(root_dir_kitti, '{:06d}.bin'.format(ref)), dtype=np.float32).reshape(-1, 4)
    xyz_ref = xyzr_ref[:, :3]
    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(xyz_ref)

    # Load pre-processed descriptor
    print("Load previously computed dip descriptor.")
    dip_current_file = os.path.join(dip_folder, '{:06d}.npz'.format(current))
    dip_ref_file = os.path.join(dip_folder, '{:06d}.npz'.format(ref))

    data_current = np.load(dip_current_file)  # previous frame
    data_ref = np.load(dip_ref_file)  # current frame

    # the indices of the sampled points whose descriptors are computed
    inds_current = data_current['inds']
    inds_ref = data_ref['inds']

    pcd_current_pts = np.asarray(pcd_current.points)[inds_current]
    pcd_ref_pts = np.asarray(pcd_ref.points)[inds_ref]

    # the descriptors of all sampled points
    pcd_current_desc = data_current['pcd1_desc']
    pcd_ref_desc = data_ref['pcd1_desc']

    # ransac
    if int(o3d.__version__.split(".")[1]) < 11:
        pcd_ref_dsdv = o3d.registration.Feature()
        pcd_current_dsdv = o3d.registration.Feature()
    else:
        pcd_ref_dsdv = o3d.pipelines.registration.Feature()
        pcd_current_dsdv = o3d.pipelines.registration.Feature()

    pcd_current_dsdv.data = pcd_current_desc.T
    pcd_ref_dsdv.data = pcd_ref_desc.T

    # fix the nan problem
    bool_nan_current = np.isnan(pcd_current_dsdv.data.sum(0))
    pcd_current_dsdv.data = pcd_current_dsdv.data[:, ~bool_nan_current]
    pcd_current_pts = pcd_current_pts[~bool_nan_current]

    bool_nan_ref = np.isnan(pcd_ref_dsdv.data.sum(0))
    pcd_ref_dsdv.data = pcd_ref_dsdv.data[:, ~bool_nan_ref]
    pcd_ref_pts = pcd_ref_pts[~bool_nan_ref]

    _pcd_current = o3d.geometry.PointCloud()
    _pcd_current.points = o3d.utility.Vector3dVector(pcd_current_pts)
    _pcd_ref = o3d.geometry.PointCloud()
    _pcd_ref.points = o3d.utility.Vector3dVector(pcd_ref_pts)

    t_exec = time.time()
    print('\tComputing ransac')
    # o3d version above 0.11.0 is faster in computing RANSAC
    if int(o3d.__version__.split(".")[1]) == 9:
        print("Using open3d version: {}".format(o3d.__version__.split(".")[1]))
        # version 9 to work on Ubuntu 16 LTS
        est_result01 = o3d.registration.registration_ransac_based_on_feature_matching(
            _pcd_current,
            _pcd_ref,
            pcd_current_dsdv,
            pcd_ref_dsdv,
            .1,
            estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(.9),
                      o3d.registration.CorrespondenceCheckerBasedOnDistance(.1)],
            criteria=o3d.registration.RANSACConvergenceCriteria(1000000, 500)
        )
    else:
        print("Using open3d version: {}".format(o3d.__version__.split(".")[1]))
        # newer version (most recent)
        est_result01 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            _pcd_current,
            _pcd_ref,
            pcd_current_dsdv,
            pcd_ref_dsdv,
            mutual_filter=False,
            max_correspondence_distance=.1,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.9),
                      o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(.1)],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 500)
        )

    print('\tFinished in {} sec'.format(time.time() - t_exec))
    T = est_result01.transformation
    fitness = est_result01.fitness
    return T, fitness


# the main function computes the transforms among candidate and ref frame,
# and save the transform, the RANSAC fitness and the DIP-based overlap to the result file
def compute_transformation(root_dir_kitti, dip_folder, pairs_json_file, json_file, pair_gap):
    if os.path.exists(json_file):
        json_data = json.load(open(json_file))
    else:
        json_data = json.load(open(pairs_json_file))
        print("No result json file, create.")

    for i, key in enumerate(json_data):
        data = json_data[key]
        if 'transformations' in data:
            print("{} is generated, skip...".format(key))
            continue
        else:
            print("Generate for {} ...".format(key))
            json_data[key]['transformations'] = []
            json_data[key]['fitness'] = []
            json_data[key]['dip_overlap'] = []
            if len(data['indices_pairs']) > 0:
                print("Pairs for {} is not none, compute for each pair".format(key))
                pairs = data['indices_pairs']
                for j, pair in enumerate(pairs):
                    if j % pair_gap == 0:
                        current_index = pair[1]
                        ref_index = pair[0]
                        pose_c2r, fitness = compute_transformation_once(root_dir_kitti, dip_folder, current_index, ref_index)
                        _, _, _, overlap = com_overlap_transform(root_dir_kitti, current_index, ref_index, pose_c2r)
                        json_data[key]["dip_overlap"].append(overlap)
                        json_data[key]['transformations'].append(pose_c2r.tolist())
                        json_data[key]['fitness'].append(fitness)
                    else:
                        print("Set Transform from current to ref as zeros")
                        json_data[key]['transformations'].append(np.zeros(4).tolist())
                        json_data[key]['fitness'].append(0)
                        json_data[key]["dip_overlap"].append(0)

                # update the json file for each frame
                print("Write for {} in the json file, only write when pair larger than 0.".format(key))
                with open(json_file, 'w') as fp:
                    json.dump(json_data, fp)


if __name__ == '__main__':
    # set the related paths
    scan_folder = os.path.join("..", "data", "kitti00", "velodyne")
    dip_folder = os.path.join("..", "data", "kitti00", "dip")
    result_folder = os.path.join("..", "data", "kitti00", "results")

    pair_gap = 1 # the indices pairs are already in every 20 candidate frames
    pairs_json_file = os.path.join(result_folder, 'overlapnet_gt_est_reloc_result_20.json')
    result_json_file = os.path.join(result_folder, 'dip_overlap_reloc_result_20.json')

    # step 1: compute the transforms between all [candidate, reference] pairs
    compute_transformation(scan_folder, dip_folder, pairs_json_file, result_json_file, pair_gap)

    # step 2: compute the RON with the estimated transforms for all [candidate, reference] pairs
    compute_mnn(scan_folder, dip_folder, result_json_file, pair_gap)