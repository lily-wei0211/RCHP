""" Collect point clouds and the corresponding labels from original S3DID dataset, and save into numpy files.


"""

import os
import glob
import numpy as np
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


def random_downsample(points, num_samples):
    """
    通过随机采样降低点云密度。

    Parameters:
    - points: np.ndarray, 原始点云数据，形状为 (n, 7)，其中每行包含 (X, Y, Z, I, R, G, B, L)
    - num_samples: int, 目标下采样后的点云数量

    Returns:
    - np.ndarray, 下采样后的点云数据
    """
    # 确保下采样后的点数不超过原始点云的点数
    if num_samples > len(points):
        raise ValueError("num_samples should not exceed the total number of points")

    # 随机选择点云的索引
    sampled_indices = np.random.choice(len(points), size=num_samples, replace=False)

    # 根据随机选中的索引获取下采样后的点云
    downsampled_points = points[sampled_indices]

    return downsampled_points


def collect_point_label(file_path,out_filename, file_format='numpy'):
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
    points = np.loadtxt(file_path)
    labels= np.loadtxt(file_path.split('.')[0] + '.labels')
    with open(file_path.split('.')[0] + '.labels', 'r') as f:
        lines = f.readlines()
    labels = [l.strip() for l in lines]
    labels = np.array(labels, dtype=float)

    # 统计出现过的label和次数

    # data_label = np.concatenate([points, labels], 1)
    data_label = np.concatenate([points, np.expand_dims(labels, axis=-1)], -1)

    # 点数太多了，随机采样处理
    sample_ratio=0.3
    data_label_sampled = random_downsample(data_label, int(len(data_label)*sample_ratio))
    print('sampled points:',data_label_sampled.shape[0])
    for i in range(3):
        print(data_label_sampled[:, i].min(), data_label_sampled[:, i].max())

    if file_format == 'txt':
        fout = open(out_filename, 'w')
        for i in range(data_label.shape[0]):
            fout.write('%f %f %f %d %d %d %d\n' % \
                       (data_label[i, 0], data_label[i, 1], data_label[i, 2],
                        data_label[i, 3], data_label[i, 4], data_label[i, 5],
                        data_label[i, 6]))
        fout.close()
    elif file_format == 'numpy':
        # np.save(out_filename, data_label)
        print(data_label_sampled.shape)
        np.save(out_filename, data_label_sampled)
    else:
        print('ERROR!! Unknown file format: %s, please use txt or numpy.' % \
              (file_format))
        exit()



class_scans={id:0 for id in range(41)}
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../datasets/Semantic3D',
                        help='Directory to dataset')
    args = parser.parse_args()


    DATA_PATH = args.data_path
    DST_PATH = os.path.join(ROOT_DIR, 'datasets/Semantic3D')
    # SAVE_PATH = os.path.join(DST_PATH, 'scenes', 'data')
    SAVE_PATH = os.path.join(DST_PATH, 'data')
    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)

    CLASS_NAMES = [x.rstrip() for x in open(os.path.join(ROOT_DIR, 'datasets/Semantic3D/meta', 'semantic3d_classnames.txt'))]
    CLASS2LABEL = {cls: i for i, cls in enumerate(CLASS_NAMES)}

    if not os.path.isdir(args.data_path):
        raise ValueError("%s does not exist" % args.data_path)

    scene_paths = sorted([fn for fn in os.listdir(args.data_path) if 'txt' in fn])

    n_scenes = len(scene_paths)
    if (n_scenes == 0):
        raise ValueError('%s is empty' % args.data_path)
    else:
        print('%d files are under this folder' % n_scenes)

    for i,scene_path in enumerate(scene_paths):
        scene_filename = os.path.join(args.data_path, scene_path)
        out_filename = os.path.join(SAVE_PATH, scene_path[:-4])+'.npy'
        try:
            if i!=6:  # lack:no.7
                continue
            print('\nBegin ({}/{}) {}'.format(i, n_scenes, scene_path))
            collect_point_label(scene_filename,out_filename)
            print('({}/{}) {}'.format(i,n_scenes,scene_path))
        except:
            print(scene_path, 'ERROR!!')

    # print('num scenes for each class:')
    # for k,v in class_scans.items():
    #     print(k,v)
    #
    # print(class_scans)

    print('finished!')
