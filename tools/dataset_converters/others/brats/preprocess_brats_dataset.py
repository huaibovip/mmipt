# Copyright (c) MMIPT. All rights reserved.
import os
import glob
import zipfile
import argparse
import os.path as osp
from multiprocessing import Pool

import medpy.io as medio
import numpy as np

import json
# from mmipt.utils import modify_args


def unzip_dataset(args):
    paths = [args.train_path, args.valid_path]
    if args.test_path is not None:
        paths.append(args.test_path)

    for path in paths:
        zip_file = zipfile.ZipFile(path)
        zip_file.extractall(args.data_root)
        zip_file.close()


def sup_128(xmin, xmax):
    if xmax - xmin < 128:
        ecart = int((128 - (xmax - xmin)) / 2)
        xmax = xmax + ecart + 1
        xmin = xmin - ecart
    if xmin < 0:
        xmax -= xmin
        xmin = 0
    return xmin, xmax


def crop(vol):
    if len(vol.shape) == 4:
        vol = np.amax(vol, axis=0)
    assert len(vol.shape) == 3

    # x_dim, y_dim, z_dim = tuple(vol.shape)
    x_nonzeros, y_nonzeros, z_nonzeros = np.where(vol != 0)

    x_min, x_max = np.amin(x_nonzeros), np.amax(x_nonzeros)
    y_min, y_max = np.amin(y_nonzeros), np.amax(y_nonzeros)
    z_min, z_max = np.amin(z_nonzeros), np.amax(z_nonzeros)

    x_min, x_max = sup_128(x_min, x_max)
    y_min, y_max = sup_128(y_min, y_max)
    z_min, z_max = sup_128(z_min, z_max)

    return x_min, x_max, y_min, y_max, z_min, z_max


def normalize(vol, method):
    mask = vol.sum(0) > 0

    if method == 'zsocre':
        for k in range(vol.shape[0]):
            x = vol[k, ...]
            y = x[mask]
            x = (x - y.mean()) / y.std()
            vol[k, ...] = x
    elif method == 'max_min':
        for k in range(vol.shape[0]):
            x = vol[k, ...]
            y = x[mask]
            x = (x - y.min()) / (y.max() - y.min())
            vol[k, ...] = x
    return vol


def load_data(imgs, label, norm_method):
    vol_list = []
    header_list = []
    for img_path in imgs:
        vol, header = medio.load(img_path)
        vol_list.append(vol)
        header_list.append(header)

    vol = np.stack(vol_list, axis=0).astype(np.float32)
    x_min, x_max, y_min, y_max, z_min, z_max = crop(vol)
    vol = normalize(vol[:, x_min:x_max, y_min:y_max, z_min:z_max], norm_method)

    if label is not None:
        seg, seg_header = medio.load(label)
        seg = seg.astype(np.uint8)
        seg = seg[x_min:x_max, y_min:y_max, z_min:z_max]
        seg[seg == 4] = 3
    else:
        seg = None

    return vol, seg


def processing(data_root, output_root, data_name, norm_method, train=False):
    if data_name in ['BRATS2018', 'BRATS2019'] and train:
        prefix = ['HGG', 'LGG']
    else:
        prefix = ['']

    output_img = osp.join(output_root, 'img')
    output_seg = osp.join(output_root, 'seg')
    os.makedirs(output_img, exist_ok=True)
    os.makedirs(output_seg, exist_ok=True)

    data_list = []
    for p in prefix:
        data_list.extend(glob.glob(osp.join(data_root, p, '*')))

    ret_data_list = []
    for data_dir in data_list:
        if not os.path.isdir(data_dir):
            continue
        case_id = osp.basename(data_dir)
        imgs = [
            glob.glob(osp.join(data_dir, case_id + '_flair.nii*'))[0],
            glob.glob(osp.join(data_dir, case_id + '_t1ce.nii*'))[0],
            glob.glob(osp.join(data_dir, case_id + '_t1.nii*'))[0],
            glob.glob(osp.join(data_dir, case_id + '_t2.nii*'))[0],
        ]
        label = glob.glob(osp.join(data_dir, case_id + '_seg.nii*'))
        if len(label) == 0:
            label = [None]

        img, seg = load_data(imgs, label[0], norm_method)

        if 'HGG' in data_dir:
            mode = 'HGG_'
        elif 'LGG' in data_dir:
            mode = 'LGG_'
        else:
            mode = ''

        save_img_path = osp.join(output_img, mode + case_id + '_img.npy')
        save_seg_path = osp.join(output_seg, mode + case_id + '_seg.npy')
        np.save(save_img_path, img)
        if seg is not None:
            np.save(save_seg_path, seg)

        ret_data_list.append(
            dict(img_path=save_img_path, seg_path=save_seg_path))
        print(img.shape)
    return ret_data_list


def generate_anno_file(data_list, file='meta_info_brats.txt'):
    """Generate anno file for Vimeo90K datasets from the official clip list.

    Args:
        clip_list (str): Clip list path for Vimeo90K datasets.
        file_name (str): Saved file name. Default: 'meta_info_Vimeo90K_GT.txt'.
    """

    data_info = dict()
    data_info['data_list'] = data_list
    data_info['metainfo'] = dict(
        classes=["Unknown", "NCR/NET", "ED", "ET"], task_name='segmentation')
    print(f'Generate annotation files {osp.basename(file)}...')
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data_info, f, ensure_ascii=False, sort_keys=False, indent=4)


def worker(args):
    """Worker for each process.

    Args:
        clip_name (str): Path of the clip.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    train_root = '/home/hb/datasets/seg/BraTS2018/MICCAI_BraTS2018_TrainingData'
    data_list = processing(train_root, osp.join(args.output_root, 'Train'),
                           args.data_name, args.norm_method)
    generate_anno_file(data_list, file='train.json')

    valid_root = '/home/hb/datasets/seg/BraTS2018/MICCAI_BraTS2018_ValidationData'
    data_list = processing(valid_root, osp.join(args.output_root, 'Val'),
                           args.data_name, args.norm_method)
    generate_anno_file(data_list, file='val.json')

    # test_root = os.path.splitext(args.test_path)[0]
    # processing(train_root, osp.join(args.data_root, 'Test'), args.norm_method)
    # generate_anno_file(data_list, file='test.json')


def parse_args():
    # modify_args()
    parser = argparse.ArgumentParser(
        description='Preprocess Vimeo90K datasets',
        epilog=
        'You can download the Vimeo90K dataset from: http://toflow.csail.mit.edu/'
    )

    path1 = '/home/hb/datasets/seg/BraTS2018/'
    path2 = '/home/hb/datasets/seg/BraTS2018/'
    parser.add_argument('--data-root', default=path1, help='dataset root')
    parser.add_argument('--output-root', default=path2, help='dataset name')
    parser.add_argument(
        '--data-name', default='BRATS2021', help='dataset name')
    parser.add_argument(
        '--norm-method',
        default='zscore',
        help='normalize method (zscore or man_min)')
    parser.add_argument(
        '--n-thread',
        nargs='?',
        default=8,
        type=int,
        help='thread number when using multiprocessing')

    args = parser.parse_args()

    args.train_path = glob.glob(osp.join(args.data_root, '*Training*.zip'))[0]
    args.valid_path = glob.glob(osp.join(args.data_root,
                                         '*Validation*.zip'))[0]
    test_path = glob.glob(osp.join(args.data_root, '*Testing*.zip'))
    if len(test_path) == 0:
        test_path = [None]
    args.test_path = test_path[0]

    return args


if __name__ == '__main__':
    args = parse_args()

    unzip_dataset(args)

    worker(args)

    # # generate image list anno file
    # generate_anno_file(
    #     osp.join(args.data_root, 'sep_trainlist.txt'),
    #     'meta_info_Vimeo90K_train_GT.txt')
    # generate_anno_file(
    #     osp.join(args.data_root, 'sep_testlist.txt'),
    #     'meta_info_Vimeo90K_test_GT.txt')
