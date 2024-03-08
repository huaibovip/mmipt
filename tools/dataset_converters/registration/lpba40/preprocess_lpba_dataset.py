import os
import json
import os.path as osp

meteinfo = dict(
    dataset_type='lpba_registration_dataset',
    task_name='registration',
    classes=[
        'Unknown', 'L-superior-frontal-gyrus', 'R-superior-frontal-gyrus',
        'L-middle-frontal-gyrus', 'R-middle-frontal-gyrus',
        'L-inferior-frontal-gyrus', 'R-inferior-frontal-gyrus',
        'L-precentral-gyrus', 'R-precentral-gyrus',
        'L-middle-orbitofrontal-gyrus', 'R-middle-orbitofrontal-gyrus',
        'L-lateral-orbitofrontal-gyrus', 'R-lateral-orbitofrontal-gyrus',
        'L-gyrus-rectus', 'R-gyrus-rectus', 'L-postcentral-gyrus',
        'R-postcentral-gyrus', 'L-superior-parietal-gyrus',
        'R-superior-parietal-gyrus', 'L-supramarginal-gyrus',
        'R-supramarginal-gyrus', 'L-angular-gyrus', 'R-angular-gyrus',
        'L-precuneus', 'R-precuneus', 'L-superior-occipital-gyrus',
        'R-superior-occipital-gyrus', 'L-middle-occipital-gyrus',
        'R-middle-occipital-gyrus', 'L-inferior-occipital-gyrus',
        'R-inferior-occipital-gyrus', 'L-cuneus', 'R-cuneus',
        'L-superior-temporal-gyrus', 'R-superior-temporal-gyrus',
        'L-middle-temporal-gyrus', 'R-middle-temporal-gyrus',
        'L-inferior-temporal-gyrus', 'R-inferior-temporal-gyrus',
        'L-parahippocampal-gyrus', 'R-parahippocampal-gyrus',
        'L-lingual-gyrus', 'R-lingual-gyrus', 'L-fusiform-gyrus',
        'R-fusiform-gyrus', 'L-insular-cortex', 'R-insular-cortex',
        'L-cingulate-gyrus', 'R-cingulate-gyrus', 'L-caudate', 'R-caudate',
        'L-putamen', 'R-putamen', 'L-hippocampus', 'R-hippocampus'
    ],
)


def write_ann(file, classes, image_names, label_names=None, test=False):
    if not test:
        data_list = list()
        for i in range(len(image_names)):
            for j in range(len(image_names)):
                if i != j:
                    if label_names is not None:
                        img_path = f'images/{image_names[i]}'
                        seg_path = f'labels/{label_names[i]}'
                        data = dict(img_path=img_path, seg_path=seg_path)
                    else:
                        src_path = f'images/{image_names[i]}'
                        dst_path = f'images/{image_names[j]}'
                        data = dict(source_path=src_path, target_path=dst_path)
                    data_list.append(data)
    else:
        data_list = list()
        for i in range(1, len(image_names)):
            if label_names is not None:
                img_path = f'images/{image_names[i]}'
                seg_path = f'labels/{label_names[i]}'
                data = dict(img_path=img_path, seg_path=seg_path)
            else:
                src_path = f'images/{image_names[i]}'
                dst_path = f'images/{image_names[0]}'
                data = dict(source_path=src_path, target_path=dst_path)
            data_list.append(data)

    data_ann = dict()
    data_ann['metainfo'] = meteinfo
    data_ann['data_list'] = data_list
    os.makedirs(os.path.dirname(file), exist_ok=True)

    print(f'Generate annotation files {osp.basename(file)}...')
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data_ann, f, ensure_ascii=False, sort_keys=False, indent=4)


if __name__ == '__main__':
    root = '/home/pu/.local/tasks/reg/mmipt/data/LPBA40_p2p'
    classes = []

    names = os.listdir(osp.join(root, 'images'))
    write_ann(
        osp.join(root, 'annotations', 'train.json'),
        classes,
        image_names=names[10:])
    write_ann(
        osp.join(root, 'annotations', 'test.json'),
        classes,
        image_names=names[:10],
        test=True)
