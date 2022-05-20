#!/usr/bin/env python3
# Developed by Yiming Wang
# Covered by the LICENSE file in the root of this project.
# Brief: This demo shows how to generate the overlap and yaw ground truth files for training and testing.

import os

import matplotlib.pyplot as plt
import numpy as np
import json


# this function obtains the loop frame index
def get_loop_index(overlaps, ind_pairs):
    if not len(overlaps) == 0:
        overlaps = np.asarray(overlaps)
        ind_pairs = np.asarray(ind_pairs, dtype=int)
        reference_idx = ind_pairs[:, 0]
        return int(np.argmax(overlaps)), int(reference_idx[np.argmax(overlaps)])
    return None, None


# this function loads the result json files
def load_result(json_file):
    if os.path.exists(json_file):
        print("\tResult json exists")
        with open(json_file) as jf:
            result_data = json.load(jf)
        return result_data
    else:
        print("\tGenerate the result file first!")
        return -1


# this function computes and plots the PR curve
def compute_pr(results_compare_list, results_method_pairs, result_file_pairs, results_data_pairs):
    # Load the GT and all method result json file
    print("Load pre-saved results for all methods")
    result_data_all = {}
    for method in result_file_pairs:
      json_file = result_file_pairs[method]
      print("\t{}".format(json_file))
      result_data_all[method] = load_result(json_file)

    max_overlap_all = {}
    gt_label_all = {}
    # Step 1: first obtain all the max overlap and the gt labels
    for frame_key in result_data_all["Overlapnet"]:
        # access the gt result
        gt_result = result_data_all["Overlapnet"][frame_key]["gt"]
        indices_pairs = result_data_all["Overlapnet"][frame_key]["indices_pairs"]

        if len(indices_pairs) > 0:
            array_idx_gt, ref_index = get_loop_index(gt_result, indices_pairs)

            # obtain the max for each method
            for res_index, result_key in enumerate(results_compare_list):
                if result_key not in max_overlap_all:
                    max_overlap_all[result_key] = []
                if result_key not in gt_label_all:
                    gt_label_all[result_key] = []
                method_key = results_method_pairs[result_key]
                data_key = results_data_pairs[result_key]
                est_result = result_data_all[method_key][frame_key][data_key]

                # print("At {} and checking result of {}, update the FP, FN, TP, TN".format(frame_key, result_key))
                if array_idx_gt is not None:
                    max_overlap_gt = gt_result[array_idx_gt]
                    array_idx_est, _ = get_loop_index(est_result, indices_pairs)
                    max_est = est_result[array_idx_est]
                    max_overlap_all[result_key].append(max_est)

                    if max_overlap_gt > overlap_thres_gt: # a true detection
                        # print("current index {} and ref index {}".format(str(idx), str(ref_index)))
                        gt_label_all[result_key].append(1)
                    else:
                        gt_label_all[result_key].append(0)

    # Step 2: obtain the pr and re numpy array
    Pr_all = {}
    Re_all = {}
    Threshold_all = {}
    for method_key in max_overlap_all:
        max_overlap_one = np.asarray(max_overlap_all[method_key])
        gt_label_one = np.asarray(gt_label_all[method_key])
        sorted_indices = np.argsort(-1 * max_overlap_one)
        gt_label_one_descending = gt_label_one[sorted_indices]
        max_overlap_one_descending = max_overlap_one[sorted_indices]

        sum_TP_FN = np.sum(gt_label_one_descending)
        pr = np.ones_like(max_overlap_one_descending)
        re = np.ones_like(max_overlap_one_descending)
        for i, max_overlap in enumerate(max_overlap_one_descending.tolist()):
            sum_TP_FP = i+1
            TP = np.sum(gt_label_one_descending[:(i+1)])
            pr[i] = TP/sum_TP_FP*100.0
            re[i] = TP/sum_TP_FN*100.0
        Pr_all[method_key] = pr.tolist()
        Re_all[method_key] = re.tolist()
        Threshold_all[method_key] = max_overlap_one_descending.tolist()
    return Pr_all, Re_all, Threshold_all


if __name__ == '__main__':
  # path to the result folder
  result_folder = os.path.join("..", "data", "kitti00", "results")

  # the threshhold for gt overlap
  overlap_thres_gt = 0.3

  results_compare_list = ["OverlapNet (repro.) [8]", "LiDAR Iris (repro.) [9]", "L3D-based Overlap", "L3D-based MNN", "L3D-based RON (ours)"]
  results_color_pairs = {"OverlapNet (repro.) [8]": "bo--", "LiDAR Iris (repro.) [9]": "g*--", "L3D-based fitness": "y^--", "L3D-based Overlap": "rh--", "L3D-based MNN": "kx--", "L3D-based RON (ours)": "md--"}
  results_method_pairs = {"OverlapNet (repro.) [8]": "Overlapnet", "LiDAR Iris (repro.) [9]": "LiDAR Iris", "L3D-based fitness": "DIP", "L3D-based Overlap": "DIP", "L3D-based MNN": "DIP", "L3D-based RON (ours)": "DIP"}
  results_data_pairs = {"OverlapNet (repro.) [8]": "overlapnet", "LiDAR Iris (repro.) [9]": "iris_score", "L3D-based fitness": "fitness", "L3D-based Overlap": "dip_overlap", "L3D-based MNN": "dip_mnn", "L3D-based RON (ours)": "dip_ron"}

  # binary to decide the PR curve of which experiment setup to visualise
  # result_reloc shows the result of the relocalisation setup
  result_reloc = False

  if result_reloc:
      result_file_pairs = {"Overlapnet": os.path.join(result_folder, "overlapnet_gt_est_reloc_result_20.json"),
                           "LiDAR Iris": os.path.join(result_folder, "iris_reloc_result_20.json"),
                           "DIP": os.path.join(result_folder, "dip_overlap_reloc_result_20.json")}
  else:
      result_file_pairs = {"Overlapnet":os.path.join(result_folder, "overlapnet_gt_est_result.json"),
                       "LiDAR Iris": os.path.join(result_folder, "iris_result.json"),
                       "DIP": os.path.join(result_folder, "dip_overlap_result.json")}

  Pr_all, Re_all, Threshold_all = compute_pr(results_compare_list, results_method_pairs, result_file_pairs, results_data_pairs)

  # visualisation
  plt.rcParams['font.size'] = '20'
  fig, ax = plt.subplots(figsize=(12, 6))

  for result_index, result_key in enumerate(results_compare_list):
      ax.plot(Re_all[result_key], Pr_all[result_key], results_color_pairs[result_key], linewidth=3, markersize = 10, markevery = 50, label= result_key)
  ax.set(xlabel = "Recall [%]")
  ax.set(ylabel = "Precision [%]")
  leg = ax.legend()

  plt.show()






