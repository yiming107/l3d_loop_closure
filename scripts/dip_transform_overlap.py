#!/usr/bin/env python3
# Developed by Yiming Wang and Fabio Poiesi
# Covered by the LICENSE file in the root of this project.
# These functions support the DIP-based transform and overlap computation.


from overlapnetutils import *
import torch
from knn_cuda import KNN


def com_overlap_transform(root_dir_kitti, current_index, ref_index, pose_c2r):
    '''
    This function computes the overlap defined in Overlapnet paper
    # using an estimated transform among a given frame and the reference frame
        root_dir_kitti: the directory of the lidar points
        current_index: the index of the current frame
        ref_index: the index of the reference frame
        pose_c2r: estimated transform from current to reference
    '''

    print('Start to compute overlap')
    current_path = os.path.join(root_dir_kitti, '{:06d}.bin'.format(current_index))
    ref_path = os.path.join(root_dir_kitti, '{:06d}.bin'.format(ref_index))
    # we calculate the ground truth for one given frame only
    # generate range projection for the given frame
    current_points = load_vertex(current_path)
    current_range, project_points, _, _ = range_projection(current_points)
    visible_points = project_points[current_range > 0]
    valid_num = len(visible_points)

    # generate range projection for the reference frame
    reference_pose = np.linalg.inv(pose_c2r)
    reference_points = load_vertex(ref_path)
    reference_points_in_current = reference_pose.dot(reference_points.T).T
    reference_range, _, _, _ = range_projection(reference_points_in_current)

    # calculate overlap
    overlap = np.count_nonzero(
      abs(reference_range[reference_range > 0] - current_range[reference_range > 0]) < 1) / valid_num
    return current_points[:,:3], reference_points_in_current[:,:3], reference_points[:,:3] , overlap


def com_overlap_mnn(dip_current, dip_ref, pcd_current, pcd_ref, posec2r, th_e):
    '''
    This function computes the overlap based on the mutual nearest neighbours in the point feature space
    using an estimated transform among a given frame and the reference frame
        dip_current: descriptors of point cloud at current index (N number of points by D descriptor dimension.)
        dip_ref: descriptors of point cloud at ref index (N number of points by D descriptor dimension.)
        pcd_current: pcd of the current index
        pcd_ref: pcd of the reference index
        posec2r: estimated transform from current to reference
        th_e: threshold of the geometric distance of points that are mutually nearest neighbour in descriptor space.
    '''

    inputcurrent_gpu = torch.tensor(dip_current, device='cuda').unsqueeze(0)
    inputref_gpu = torch.tensor(dip_ref, device='cuda').unsqueeze(0)

    knn = KNN(k=1, transpose_mode=True)

    _, nn_ref_inds_gpu = knn(inputref_gpu, inputcurrent_gpu)
    _, nn_current_inds_gpu = knn(inputcurrent_gpu, inputref_gpu)

    nn_current_inds = nn_current_inds_gpu.cpu().numpy().squeeze()
    nn_ref_inds = nn_ref_inds_gpu.cpu().numpy().squeeze()

    mnn_bools = list(range(dip_current.shape[0])) == nn_current_inds[nn_ref_inds]

    mnn_ratio = mnn_bools.sum() / np.min([len(dip_current), len(dip_ref)])

    pcd_ref.transform(np.linalg.inv(posec2r))

    dists = np.linalg.norm(np.asarray(pcd_current.points) - np.asarray(pcd_ref.points)[nn_ref_inds], axis=1)

    ron = (dists[mnn_bools] < th_e).sum() / mnn_bools.sum()

    return mnn_ratio, ron
