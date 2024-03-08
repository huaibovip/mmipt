# Copyright (c) OpenMMLab. All rights reserved.
from .albumentations import Albumentations
from .alpha import GenerateSeg, GenerateSoftSeg
from .aug_frames import MirrorSequence, TemporalReverse
from .aug_pixel import (
    BinarizeImage,
    Clip,
    ColorJitter,
    RandomAffine,
    RandomMaskDilation,
    UnsharpMasking,
)
from .aug_shape import (
    Flip,
    NumpyPad,
    RandomRotation,
    RandomTransposeHW,
    Resize,
)
from .crop import (
    CenterCropLongEdge,
    Crop,
    CropAroundCenter,
    CropAroundFg,
    CropAroundUnknown,
    CropLike,
    FixedCrop,
    InstanceCrop,
    ModCrop,
    PairedRandomCrop,
    RandomCropLongEdge,
    RandomResizedCrop,
)
from .fgbg import (
    CompositeFg,
    MergeFgAndBg,
    PerturbBg,
    RandomJitter,
    RandomLoadResizeBg,
)
from .flip import RandomFlip
from .formatting import PackInputs, InjectMeta
from .generate_assistant import (
    GenerateCoordinateAndCell,
    GenerateFacialHeatmap,
)
from .generate_frame_indices import (
    GenerateFrameIndices,
    GenerateFrameIndiceswithPadding,
    GenerateSegmentIndices,
)
from .get_masked_image import GetMaskedImage
from .loading import (
    GetSpatialDiscountMask,
    LoadImageFromFile,
    LoadMask,
    LoadPairedImageFromFile,
    LoadVolumeFromFile,
    LoadBundleVolumeFromFile,
    LoadQuadVolumeFromFile,
)
from .matlab_like_resize import MATLABLikeResize
from .normalization import Normalize, RescaleToZeroOne
from .random_degradations import (
    DegradationsWithShuffle,
    RandomBlur,
    RandomJPEGCompression,
    RandomNoise,
    RandomResize,
    RandomVideoCompression,
)
from .random_down_sampling import RandomDownSampling
from .segmentation import (
    SegmentNormalize,
    IXISegmentNormalize,
    LPBASegmentNormalize,
)
from .trimap import (
    FormatTrimap,
    GenerateTrimap,
    GenerateTrimapWithDistTransform,
    TransformTrimap,
)
from .values import CopyValues, SetValues

__all__ = [
    'BinarizeImage', 'Clip', 'ColorJitter', 'CopyValues', 'Crop', 'CropLike',
    'DegradationsWithShuffle', 'LoadImageFromFile', 'LoadMask', 'Flip',
    'FixedCrop', 'GenerateCoordinateAndCell', 'GenerateFacialHeatmap',
    'GenerateFrameIndices', 'GenerateFrameIndiceswithPadding',
    'GenerateSegmentIndices', 'GetMaskedImage', 'GetSpatialDiscountMask',
    'MATLABLikeResize', 'MirrorSequence', 'ModCrop', 'Normalize', 'PackInputs',
    'InjectMeta', 'PairedRandomCrop', 'RandomAffine', 'RandomBlur',
    'RandomDownSampling', 'RandomJPEGCompression', 'RandomMaskDilation',
    'RandomNoise', 'RandomResize', 'RandomResizedCrop', 'RandomRotation',
    'RandomTransposeHW', 'RandomVideoCompression', 'RescaleToZeroOne',
    'Resize', 'SetValues', 'TemporalReverse', 'ToTensor', 'UnsharpMasking',
    'CropAroundCenter', 'CropAroundFg', 'GenerateSeg', 'CropAroundUnknown',
    'GenerateSoftSeg', 'FormatTrimap', 'TransformTrimap', 'GenerateTrimap',
    'GenerateTrimapWithDistTransform', 'CompositeFg', 'RandomLoadResizeBg',
    'MergeFgAndBg', 'PerturbBg', 'RandomJitter', 'LoadPairedImageFromFile',
    'RandomFlip', 'SegmentNormalize', 'IXISegmentNormalize',
    'LPBASegmentNormalize', 'LoadVolumeFromFile', 'LoadBundleVolumeFromFile',
    'LoadQuadVolumeFromFile', 'CenterCropLongEdge', 'RandomCropLongEdge',
    'NumpyPad', 'InstanceCrop', 'Albumentations'
]
