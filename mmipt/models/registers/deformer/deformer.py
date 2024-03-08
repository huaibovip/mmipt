import torch
import torch.nn as nn
import torch.nn.functional as F

from . import configs
from .module import Deformer, DeformableSkipLearner


def conv_block(dim,
               in_channels,
               out_channels,
               kernel_size=3,
               stride=1,
               padding=1,
               batchnorm=False):
    conv_fn = getattr(nn, "Conv{0}d".format(dim))
    bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
    if batchnorm:
        layer = nn.Sequential(
            conv_fn(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding), bn_fn(out_channels), nn.LeakyReLU(0.2))
    else:
        layer = nn.Sequential(
            conv_fn(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding), nn.LeakyReLU(0.2))
    return layer


class Encoder(nn.Module):
    '''
    Encoder of DMR, 4 3D conv layer for 1/2, 1/4, 1/8, 1/16 scale
        :param dim: the dimension of input [batch,dim,d,w,h].
        :param bn: whether use batch normalization or not, True->use.
        :param x: the input medical image [batch,dim,d,w,h].
        :return x: feature at 1/16 scale.
        :return x_enc: feature at 1/2, 1/4, 1/8, 1/16 scale.
    '''

    def __init__(self, dim, bn=True):
        super(Encoder, self).__init__()
        self.bn = bn
        self.dim = dim
        self.enc_nf = [16, 32, 32, 64]
        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(self.enc_nf)):
            prev_nf = 1 if i == 0 else self.enc_nf[i - 1]
            self.enc.append(
                conv_block(dim, prev_nf, self.enc_nf[i], 4, 2, batchnorm=bn))

    def forward(self, x):
        # Get encoder activations
        x_enc = [x]
        for i, l in enumerate(self.enc):
            x = l(x_enc[-1])
            x_enc.append(x)
        return x, x_enc


class DMR(nn.Module):
    '''
    Implementation details of DMR:
        :param dim: the dimension of input [batch,dim,d,w,h].
        :param vol: the size of medical image in LPBA40 [160,192,160] or OASIS [224,192,160]
        :param layer: From coarse to fine, decide which layers use Deformer. Otherwise use concatenation.
        if layer=2, means 1/16 and 1/8 resolution use Deformer. 1/4 and 1/2 use concatenation.
        :return flow: displacement field at full resolution [batch,3,d,w,h].
        :return corrs: sub displacement field at 1/2,1/4,1/8,1/16 resolution.
    '''

    def __init__(self, dim, vol, layer):
        super(DMR, self).__init__()
        # One conv to get the flow field
        self.vol = vol
        self.layer = layer
        self.backbone = Encoder(dim)
        self.derlearn = DeformableSkipLearner([3, 3, 3, 3])
        self.derlayer = nn.ModuleList()

        d, w, h = vol
        d = d // 32
        w = w // 32
        h = h // 32

        # Deformer
        self.derlayer.append(
            Deformer(
                n_layers=1,
                d_model=64,
                n_position=d * w * h * 8,
                n_head=8,
                n_point=64))
        self.derlayer.append(
            Deformer(
                n_layers=1,
                d_model=32,
                n_position=d * w * h * 64,
                n_head=8,
                n_point=64))
        self.derlayer.append(
            Deformer(
                n_layers=1,
                d_model=32,
                n_position=d * w * h * 512,
                n_head=8,
                n_point=64))
        self.derlayer.append(
            Deformer(
                n_layers=1,
                d_model=16,
                n_position=d * w * h * 4096,
                n_head=8,
                n_point=64))
        # Deformer->VIT
        '''
        self.config_vit = configs.get_3DReg_config()
        self.derlayer.append(
            ViTVNet(self.config_vit, in_channels=64, img_size=(2*d, 2*w, 2*h)))
        self.derlayer.append(
            ViTVNet(self.config_vit, in_channels=32, img_size=(4*d, 4*w, 4*h)))
        self.derlayer.append(
            ViTVNet(self.config_vit, in_channels=32, img_size=(8*d, 8*w, 8*h)))
        self.derlayer.append(
            ViTVNet(self.config_vit, in_channels=16, img_size=(16*d, 16*w, 16*h)))
        '''
        # Deformer->3D ResNet
        '''
        self.derlayer.append(
            generate_model(50, n_input_channels=64*2))
        self.derlayer.append(
            generate_model(50, n_input_channels=32*2))
        self.derlayer.append(
            generate_model(50, n_input_channels=32*2))
        self.derlayer.append(
            generate_model(50, n_input_channels=16*2))
        '''

    def forward(self, source, target, **kwargs):
        input = torch.cat([source, target], dim=0)
        _, feature = self.backbone(input)
        b = source.shape[0]
        moving_feature = []
        fixed_feature = []
        corrs = []

        for i in feature:
            moving_feature.append(i[0:b])
            fixed_feature.append(i[b:])

        for i in range(1, self.layer + 1):
            b, c, d, w, h = moving_feature[-i].size()
            moving_feat = moving_feature[-i].clone().flatten(2).transpose(1, 2)
            fixed_feat = fixed_feature[-i].clone().flatten(2).transpose(1, 2)
            corr = self.derlayer[i - 1](moving_feat, fixed_feat)
            corr = corr.transpose(1, 2).view(b, 3, d, w, h)
            corrs.append(corr)

        for i in range(self.layer + 1, 5):
            corrs.append(
                torch.cat([moving_feature[-i], fixed_feature[-i]], dim=1))

        flow = self.derlearn(corrs, moving_feature, fixed_feature)
        return flow, corrs


CONFIGS = {
    "Deformer": configs.get_3DReg_config(),
}
