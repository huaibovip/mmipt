# from mmcv.cnn import build_conv_layer, ConvModule

img_size = (160, 192, 224)
nb_features = [
    [16, 32, 32, 32],  # encoder
    [32, 32, 32, 32, 32, 16, 16]  # decoder
]

# in_channels = nb_features[1][-1]
# out_channels = len(img_size)
# kernel_size = 5
# stride = 1
# conv_padding = 0
# dilation = 1
# groups = 1
# bias = True

# conv_cfg = dict(
#     type=f'Conv{len(img_size)}d',
#     in_channels=nb_features[1][-1],
#     out_channels=len(img_size),
#     kernel_size=3,
#     padding=1,
#     bias=True,
# )

# # conv = build_conv_layer(
# #     conv_cfg,
# #     in_channels,
# #     out_channels,
# #     kernel_size,
# #     stride=stride,
# #     padding=conv_padding,
# #     dilation=dilation,
# #     groups=groups,
# #     bias=bias)

# conv = ConvModule(
#     in_channels,
#     out_channels,
#     kernel_size,
#     stride=stride,
#     padding=conv_padding,
#     dilation=dilation,
#     groups=groups,
#     bias=bias,
#     conv_cfg=dict(type=f'Conv{len(img_size)}d'),
#     norm_cfg=None,
#     act_cfg=None,
# )

# print(conv)

from mmengine.registry import MODELS, count_registered_modules

count_registered_modules()

conv = dict(
    type='ConvModule',
    in_channels=nb_features[1][-1],
    out_channels=len(img_size),
    kernel_size=3,
    padding=1,
    bias=True,
    conv_cfg=dict(type=f'Conv{len(img_size)}d'),
    norm_cfg=None,
    act_cfg=None)

conv = MODELS.build(conv)
print(conv)
