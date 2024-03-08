# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.utils import is_str


def ixi45_classes():
    """IXI class names for external use."""
    return [
        '#Unknown', 'Left-Cerebral-White-Matter', 'Left-Cerebral-Cortex',
        'Left-Lateral-Ventricle', '#Left-Inf-Lat-Vent',
        'Left-Cerebellum-White-Matter', 'Left-Cerebellum-Cortex',
        'Left-Thalamus-Proper*', 'Left-Caudate', 'Left-Putamen',
        'Left-Pallidum', '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem',
        'Left-Hippocampus', 'Left-Amygdala', 'CSF', '#Left-Accumbens-area',
        'Left-VentralDC', '#Left-vessel', 'Left-choroid-plexus',
        'Right-Cerebral-White-Matter', 'Right-Cerebral-Cortex',
        'Right-Lateral-Ventricle', '#Right-Inf-Lat-Vent',
        'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex',
        'Right-Thalamus-Proper*', 'Right-Caudate', 'Right-Putamen',
        'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala',
        '#Right-Accumbens-area', 'Right-VentralDC', '#Right-vessel',
        'Right-choroid-plexus', '#5th-Ventricle', '#WM-hypointensities',
        '#non-WM-hypointensities', '#Optic-Chiasm', '#CC_Posterior',
        '#CC_Mid_Posterior', '#CC_Central', '#CC_Mid_Anterior', '#CC_Anterior'
    ]


def ixi45_palette():
    """IXI palette for external use."""
    return [[0, 0, 0], [245, 245, 245], [205, 62, 78], [120, 18, 134],
            [196, 58, 250], [220, 248, 164], [230, 148, 34], [0, 118, 14],
            [122, 186, 220], [236, 13, 176], [12, 48, 255], [204, 182, 142],
            [42, 204, 164], [119, 159, 176], [220, 216, 20], [103, 255, 255],
            [60, 60, 60], [255, 165, 0], [165, 42, 42], [160, 32, 240],
            [0, 200, 200], [245, 245, 245], [205, 62, 78], [120, 18, 134],
            [196, 58, 250], [220, 248, 164], [230, 148, 34], [0, 118, 14],
            [122, 186, 220], [236, 13, 176], [13, 48, 255], [220, 216, 20],
            [103, 255, 255], [255, 165, 0], [165, 42, 42], [160, 32, 240],
            [0, 200, 221], [120, 190, 150], [200, 70, 255], [164, 108, 226],
            [234, 169, 30], [0, 0, 64], [0, 0, 112], [0, 0, 160], [0, 0, 208],
            [0, 0, 255]]


def ixi_classes():
    """IXI class names for external use."""
    return [
        "Unknown", "Left-Cerebral-White-Matter", "Left-Cerebral-Cortex",
        "Left-Lateral-Ventricle", "Left-Cerebellum-White-Matter",
        "Left-Cerebellum-Cortex", "Left-Thalamus-Proper*", "Left-Caudate",
        "Left-Putamen", "Left-Pallidum", "3rd-Ventricle", "4th-Ventricle",
        "Brain-Stem", "Left-Hippocampus", "Left-Amygdala", "CSF",
        "Left-VentralDC", "Left-choroid-plexus", "Right-Cerebral-White-Matter",
        "Right-Cerebral-Cortex", "Right-Lateral-Ventricle",
        "Right-Cerebellum-White-Matter", "Right-Cerebellum-Cortex",
        "Right-Thalamus-Proper*", "Right-Caudate", "Right-Putamen",
        "Right-Pallidum", "Right-Hippocampus", "Right-Amygdala",
        "Right-VentralDC", "Right-choroid-plexus"
    ]


def ixi_palette():
    """IXI palette for external use."""
    return [[0, 0, 0], [245, 245, 245], [205, 62, 78], [120, 18, 134],
            [220, 248, 164], [230, 148, 34], [0, 118, 14], [122, 186, 220],
            [236, 13, 176], [12, 48, 255], [204, 182, 142], [42, 204, 164],
            [119, 159, 176], [220, 216, 20], [103, 255, 255], [60, 60, 60],
            [165, 42, 42], [0, 200, 200], [245, 245, 245], [205, 62, 78],
            [120, 18, 134], [220, 248, 164], [230, 148, 34], [0, 118, 14],
            [122, 186, 220], [236, 13, 176], [13, 48, 255], [220, 216, 20],
            [103, 255, 255], [165, 42, 42], [0, 200, 221]]


def oasis_classes():
    """OASIS class names for external use."""
    return [
        "Unknown", "Left-Cerebral-White-Matter", "Left-Cerebral-Cortex",
        "Left-Lateral-Ventricle", "Left-Inf-Lat-Ventricle",
        "Left-Cerebellum-White-Matter", "Left-Cerebellum-Cortex",
        "Left-Thalamus", "Left-Caudate", "Left-Putamen", "Left-Pallidum",
        "3rd-Ventricle", "4th-Ventricle", "Brain-Stem", "Left-Hippocampus",
        "Left-Amygdala", "Left-Accumbens", "Left-Ventral-DC", "Left-Vessel",
        "Left-Choroid-Plexus", "Right-Cerebral-White-Matter",
        "Right-Cerebral-Cortex", "Right-Lateral-Ventricle",
        "Right-Inf-Lat-Ventricle", "Right-Cerebellum-White-Matter",
        "Right-Cerebellum-Cortex", "Right-Thalamus", "Right-Caudate",
        "Right-Putamen", "Right-Pallidum", "Right-Hippocampus",
        "Right-Amygdala", "Right-Accumbens", "Right-Ventral-DC",
        "Right-Vessel", "Right-Choroid-Plexus"
    ]


def oasis_palette():
    """OASIS palette for external use."""
    return [[0, 0, 0], [245, 245, 245], [205, 62, 78], [120, 18, 134],
            [196, 58, 250], [220, 248, 164], [230, 148, 34], [0, 118, 14],
            [122, 186, 220], [236, 13, 176], [12, 48, 255], [204, 182, 142],
            [42, 204, 164], [119, 159, 176], [220, 216, 20], [103, 255, 255],
            [255, 165, 0], [165, 42, 42], [160, 32, 240], [0, 200, 200],
            [245, 245, 245], [205, 62, 78], [120, 18, 134], [196, 58, 250],
            [220, 248, 164], [230, 148, 34], [0, 118, 14], [122, 186, 220],
            [236, 13, 176], [12, 48, 255], [220, 216, 20], [103, 255, 255],
            [255, 165, 0], [165, 42, 42], [160, 32, 240], [0, 200, 200]]


dataset_aliases = {
    'ixi45': ['ixi45'],
    'ixi': ['ixi', 'ixi30'],
    'oasis': ['oasis', 'oasis35'],
}


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels


def get_palette(dataset):
    """Get class palette (RGB) of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_palette()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels
