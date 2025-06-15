""" Collect point clouds and the corresponding labels from original S3DID dataset, and save into numpy files.


"""

import os
import glob
import numpy as np
import sys
import h5py
from preprocess.NYU_Depth_V2_depth2pc import *
from scipy.io import loadmat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


def collect_point_label(points, rgb, labels,out_filename, file_format='numpy'):
    """ Convert original dataset files to data_label file (each line is XYZRGBL).
        We aggregated all the points from each instance in the room.
    Args:
        anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename: path to save collected points and labels (each line is XYZRGBL)
        file_format: txt or numpy, determines what file format to save.
    Returns:
        None
    Note:
        the points are shifted before save, the most negative point is now at origin.
    """

    # 统计出现过的label和次数
    # data_label = np.concatenate([points, rgb, np.expand_dims(labels, axis=-1)], -1)
    data_label = np.concatenate([points, rgb, labels], -1)

    # for id in set(labels):
    #     class_scans[id] = class_scans[id] + 1

    if file_format == 'txt':
        fout = open(out_filename, 'w')
        for i in range(data_label.shape[0]):
            fout.write('%f %f %f %d %d %d %d\n' % \
                       (data_label[i, 0], data_label[i, 1], data_label[i, 2],
                        data_label[i, 3], data_label[i, 4], data_label[i, 5],
                        data_label[i, 6]))
        fout.close()
    elif file_format == 'numpy':
        np.save(out_filename, data_label)
        print('save:', out_filename)
    else:
        print('ERROR!! Unknown file format: %s, please use txt or numpy.' % \
              (file_format))
        exit()

# def extract_labels(labels):
#     from scipy.io import loadmat
#     label40 = loadmat('labels40.mat')['labels40']
#     mapping40 = loadmat('classMapping40.mat')['mapClass'][0]
#     name40=[name[0] for name in loadmat('classMapping40.mat')['className'][0]]
#     # mapping13 = loadmat('class13Mapping.mat')['classMapping13'][0][0][0][0]
#     mapping40 = np.insert(mapping40, 0, 0)
#     # mapping13 = np.insert(mapping13, 0, 0)
#     labels = labels.transpose([0, 2, 1])
#     labels_40 = mapping40[labels]
#     # labels_13 = mapping13[labels_40].astype('uint8')
#     return labels_40


class_scans={id:0 for id in range(41)}

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../datasets/nyu_depth_v2_labeled.mat.1',help='Directory to dataset')
    args = parser.parse_args()

    f = h5py.File(args.data_path, 'r')
    rgbset = np.array(f['images']).transpose(0, 3, 2, 1)  # (1449, 3, 640, 480)->(1449, 480, 640, 3)
    depthset = np.array(f['depths']).transpose(0, 2, 1) # (1449, 640, 480) -> (1449, 480, 640)
    # labelset = np.array(f['labels'])
    labelset40 = loadmat('labels40.mat')['labels40'].transpose(2, 0, 1)  # (480, 640, 1449) -> (1449, 480, 640)
    class_names = ['unknow', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator', 'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']

    DST_PATH = os.path.join(ROOT_DIR, 'datasets/NYUDepthV2')
    # SAVE_PATH = os.path.join(DST_PATH, 'scenes', 'data')
    SAVE_PATH = os.path.join(DST_PATH, 'data')
    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)

    # n_scenes = 1449
    n_scenes = rgbset.shape[0]

    for i in range(n_scenes):
        out_filename = os.path.join(SAVE_PATH, str(i))
        try:
            # scale = 1000.0
            scale = 1.0
            sample = convert_rgbd2pc(rgbset[i], depthset[i], labelset40[i], scale)
            # sample = {'coord': scaled_depth_PC, 'color': reshaped_rgb, 'semantic_gt': label_GT, 'normal': reshpaed_normal}
            collect_point_label(sample['coord'],sample['color'],sample['semantic_gt'],out_filename)
        except:
            print(out_filename, 'ERROR!!')

    print('num scenes for each class:')
    for k,v in class_scans.items():
        print(k,v)

    print(class_scans)

    print('finished!')
