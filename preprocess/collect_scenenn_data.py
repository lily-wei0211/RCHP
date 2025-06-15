""" Collect point clouds and the corresponding labels from original S3DID dataset, and save into numpy files.


"""

import os
import glob
import numpy as np
import sys
import h5py


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


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
    with h5py.File(file_path, 'r') as f:
        points = np.array(f['data'])
        labels = np.array(f['label'])

    # 统计出现过的label和次数

    # data_label = np.concatenate([points, labels], 1)
    data_label = np.concatenate([points, np.expand_dims(labels, axis=-1)], -1)

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
        for i in range(points.shape[0]):
            if data_label[i].shape[0]!=4096 or data_label[i].shape[1]!=13:
                continue
            out_filename_i = out_filename+'_'+str(i)+'.npy'
            for id in set(labels[i]):
                class_scans[id]=class_scans[id]+1
            # print(out_filename_i)
            # np.save(out_filename_i, data_label[i])
    else:
        print('ERROR!! Unknown file format: %s, please use txt or numpy.' % \
              (file_format))
        exit()

class_scans={id:0 for id in range(41)}

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', default='datasets/S3DIS/Stanford3dDataset_v1.2_Aligned_Version',help='Directory to dataset')
    # parser.add_argument('--data_path', default='/data/wlili/3Dseg/datasets/Stanford3dDataset_v1.2',help='Directory to dataset')
    parser.add_argument('--data_path', default='../datasets/scenenn_seg',help='Directory to dataset')
    args = parser.parse_args()


    DST_PATH = os.path.join(ROOT_DIR, 'datasets/SceneNN')
    # SAVE_PATH = os.path.join(DST_PATH, 'scenes', 'data')
    SAVE_PATH = os.path.join(DST_PATH, 'data')
    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)

    CLASS_NAMES = [x.rstrip() for x in open(os.path.join(ROOT_DIR, 'datasets/SceneNN/meta', 'scenenn_classnames.txt'))]
    CLASS2LABEL = {cls: i for i, cls in enumerate(CLASS_NAMES)}

    if not os.path.isdir(args.data_path):
        raise ValueError("%s does not exist" % args.data_path)

    # all the scenes in current Area
    scene_paths = [fn for fn in os.listdir(args.data_path) if 'seg' in fn]

    n_scenes = len(scene_paths)
    if (n_scenes == 0):
        raise ValueError('%s is empty' % args.data_path)
    else:
        print('%d files are under this folder' % n_scenes)

    for i,scene_path in enumerate(scene_paths):
        scene_filename = os.path.join(args.data_path, scene_path)
        out_filename = os.path.join(SAVE_PATH, scene_path[:-5]) 
        try:
            collect_point_label(scene_filename,out_filename)
            print('({}/{}) {}'.format(i,n_scenes,scene_path))
        except:
            print(scene_path, 'ERROR!!')

    print('num scenes for each class:')
    for k,v in class_scans.items():
        print(k,v)

    print(class_scans)

    print('finished!')
