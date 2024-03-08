# Copyright (c) MMIPT. All rights reserved.
import argparse
from collections import OrderedDict

import torch
from packaging import version


def parse_args():
    parser = argparse.ArgumentParser(description='Process a checkpoint')
    parser.add_argument('--src_path', help='input checkpoint path')
    parser.add_argument('--dst_path', help='output checkpoint path')
    args = parser.parse_args()
    return args


def delete_keys_func(state_dict, delete_keys):
    # remove delete_keys for smaller file size
    for k in list(state_dict.keys()):
        for delete_key in delete_keys:
            if k.find(delete_key) != -1:
                del state_dict[k]


def convert_state_dict(state_dict, delete_keys):
    delete_keys_func(state_dict, delete_keys)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.find('reg_head.0') != -1:
            new_k = k.replace('reg_head.0', 'flow.conv')
            new_state_dict[new_k] = v
        else:
            new_state_dict['backbone.' + k] = v

    return new_state_dict


def save_checkpoint(checkpoint, path):
    if version.parse(torch.__version__) >= version.parse('1.6'):
        torch.save(checkpoint, path, _use_new_zipfile_serialization=False)
    else:
        torch.save(checkpoint, path)


def print_state_dict(state_dict):
    for k, v in state_dict.items():
        print(f'{k}: {v.shape}')


if __name__ == '__main__':
    args = parse_args()

    args.src_path = '/home/pu/.local/tasks/reg/TransMorphLarge_Validation_dsc0.8623.pth.tar'
    args.dst_path = 'transmorph-large_oasis.pth'
    delete_keys = ['spatial_trans', 'c2.0']

    # args.src_path = '/home/pu/.local/tasks/reg/TransMorph_Validation_dsc0.744.pth.tar'
    # args.dst_path = 'transmorph_ixi.pth'
    # delete_keys = ['spatial_trans']

    ckpt = torch.load(args.src_path, map_location='cpu')

    state_dict = ckpt['state_dict']
    print_state_dict(state_dict)

    new_state_dict = convert_state_dict(state_dict, delete_keys)
    ckpt['state_dict'] = new_state_dict
    save_checkpoint(ckpt, args.dst_path)
    print_state_dict(new_state_dict)
