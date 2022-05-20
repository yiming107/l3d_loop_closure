# Loop closure detection using local 3D deep descriptors

We propose a simple yet effective method to address loop closure detection in simultaneous localisation and mapping using local 3D deep descriptors (L3Ds). L3Ds are emerging compact representations of patches extracted from point clouds that are learnt from data using a deep learning algorithm.
We propose a novel overlap measure for loop detection by computing the metric error between points that correspond to mutually-nearest-neighbour descriptors after registering the loop candidate point cloud by its estimated relative pose.
This novel approach enables us to accurately detect loops and estimate six degrees-of-freedom poses in the case of small overlaps.
We compare our L3D-based loop closure approach with recent approaches on LiDAR data and achieve state-of-the-art loop closure detection accuracy.
Additionally, we embed our loop closure approach in RESLAM, a recent edge-based SLAM system, and perform the evaluation on real-world RGBD-TUM and synthetic ICL datasets. Our approach enables RESLAM to achieve a better localisation accuracy compared to its original loop closure strategy.

# Code
Please clone this repository onto your machine first and follow the indications below to 
reproduce the results of LiDAR-based loop closure detection that are reported in [our paper](https://arxiv.org/abs/2111.00440).

## Setup
Conda environment is encouraged to be used to set up the project. You can create the conda environment
using the provided *environment.yml* file:

`conda env create -f environment.yml`

`conda activate l3d_lcd`

## Data
You can download the test sequence of Kitti 00 and the pre-computed results and metadata by runing:

`cd scripts`

`python3 download_data.py`

This will download all needed data and arrange it within your project directory, under the */data* folder.

## Reproduce the PR curves
We provide within the */data/results*, all the pre-saved results in json format, including the result of [LiDAR Iris](https://github.com/BigMoWangying/LiDAR-Iris)
and [OverlapNet](https://github.com/PRBonn/OverlapNet) by exploiting their provided repo under two setups: 1) the loop closure detection setup as described in 
[OverlapNet paper](https://arxiv.org/abs/2105.11344) and 2) the relocalisation setup as described in [our paper](https://arxiv.org/abs/2111.00440). You can reproduce the PR curve
as reported in Fig.2 and Fig. 3 of our paper, respectively by running:

`python3 compute_PRcurve.py`

Optionally, you can also reproduce the result data of our DIP-based overlap computation for loop closure detection
and relocalisation, i.e. the two experimental setups, by running:

`python3 exp_dip_overlap_kitti.py`

`python3 exp_reloc_dip_overlap_kitti.py`

Note that RANSAC-based transform estimation can be the bottleneck of the computational speed. 
For the results of the compared methods, please refer to their corresponding repo and
map the obtained results into our required json format.

# Citation
If you find our work useful in your research, please consider citing:
> @articles{zhou2021ron,\
  author={Zhou, Youjie and Wang, Yiming and Poiesi, Fabio and Qin, Qi and Wan, Yi},
  journal={IEEE Robotics and Automation Letters}, 
  title={Loop Closure Detection Using Local 3D Deep Descriptors}, 
  year={2022},
  volume={7},
  number={3},
  pages={6335-6342},
  doi={10.1109/LRA.2022.3156940}}
}
