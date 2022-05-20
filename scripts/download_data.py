#!/usr/bin/env python3
# Developed by Fabio Poiesi and Yiming Wang
# Covered by the LICENSE file in the root of this project.
# This file automatically download all the data (the test Kitti sequence and pre-saved results) into the project folder

import os
import gdown


def download_unzip(dest_dir, dataset_name, url):
    if not os.path.isdir(os.path.join(dest_dir, dataset_name)):
        output = os.path.join(dest_dir, '%s.zip' % dataset_name)
        gdown.download(url, output, fuzzy=True)
        cmd = 'unzip %s -d %s' % (output, dest_dir)
        os.system(cmd)
        cmd = 'rm %s' % output
        os.system(cmd)
    else:
        print('[i] directory <%s> already existing. Dataset <%s> will not be downloaded' % (dest_dir, dataset_name))


if __name__ == '__main__':
    dest_dir = os.path.join('..', 'data')
    os.makedirs(dest_dir, exist_ok=True)

    download_unzip(dest_dir= dest_dir,
                   dataset_name='kitti00',
                   url='https://drive.google.com/file/d/1fUnforKuRx0SmvMExaiZ2owwEDqnkSed/view?usp=sharing')