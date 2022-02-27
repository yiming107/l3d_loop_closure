# Loop closure detection using local 3D deep descriptors (RAL 2022)

In this work, we propose a simple yet effective method to address loop closure detection in simultaneous localisation and mapping using local 3D deep descriptors (L3Ds). L3Ds are emerging compact representations of patches extracted from point clouds that are learnt from data using a deep learning algorithm.
We propose a novel overlap measure for loop detection by computing the metric error between points that correspond to mutually-nearest-neighbour descriptors after registering the loop candidate point cloud by its estimated relative pose.
This novel approach enables us to accurately detect loops and estimate six degrees-of-freedom poses in the case of small overlaps.
We compare our L3D-based loop closure approach with recent approaches on LiDAR data and achieve state-of-the-art loop closure detection accuracy.
Additionally, we embed our loop closure approach in RESLAM, a recent edge-based SLAM system, and perform the evaluation on real-world RGBD-TUM and synthetic ICL datasets. Our approach enables RESLAM to achieve a better localisation accuracy compared to its original loop closure strategy.

# Code will come soon!

# Citation
If you find our work useful in your research, please consider citing:
> @articles{zhou2021ron,\
   author = {Zhou, Youjie and Wang, Yiming and Poiesi, Fabio and Qin, Qi and Wan, Yi},\
   titile = {Loop closure detection using local 3D deep descriptors},\
   journal = {IEEE Robotics and Automation Letters},\
   year = {to appear in 2022}
}
