import torch
import torch.nn as nn
import torch.nn.functional as nnf
from einops.layers.torch import Rearrange

from mmipt.registry import MODELS
from .vit_transformer import (
    PatchEmbed,
    PatchExpanding,
    SkipConnection_v2,
    VisionTransformer,
)


class SpatialTransform(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size):
        super().__init__()

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid, persistent=False)

    def forward(self, src, flow, mode='bilinear'):

        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (
                new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2**self.nsteps)
        self.transformer = SpatialTransform(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)

        # vec = torch.exp(vec)
        return vec


class UpSample(nn.Module):

    def __init__(self, in_chans, out_chans, bias=False):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.deconv = nn.ConvTranspose3d(
            in_channels=self.in_chans,
            out_channels=self.out_chans,
            kernel_size=2,
            stride=2,
            bias=bias)

    def forward(self, x):
        x = self.deconv(x)
        return x


class Reshape(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.reshape = nn.Sequential(
            Rearrange('B C D H W -> B (D H W) C'),
            nn.LayerNorm(embed_dim, eps=1e-6))

    def forward(self, x):
        reshaped_x = self.reshape(x)
        return reshaped_x


class RegTran(nn.Module):

    def __init__(self, img_size, base_channel, down_ratio, patch_size, n_heads,
                 sr_ratio):
        """
        Vit for registration with some progress.
        :param feature_shape: tuple
            Feature maps shape where the feature maps firstly input the encoder vit.
        :param C: int
            The constant represent the dimension output from the Vit.
        :param vit_depth: int
            Each level the transformer depth.
        :param patch_size: int
            Patch size through the whole model.
        :param n_heads: int
            Number of heads.


        -------------
        Attributes:
        patch_emb:
            Embedding patches.
        ds:
            down sample to get the low resolution image(tensor)
        vit:
            Vision Transformation block.
        norm:
            out put deformation field. Range[-1,1]
        """
        super().__init__()

        assert list(patch_size) == [
            patch_size[0]
        ] * len(patch_size), 'Patch size must be squared.'

        self.stride = 2
        # self.learning_mode = learning_mode
        self.pad = ((patch_size[0] - self.stride) // 2) + 1

        # Level 0: 1/1*origin_shape
        self.encoder_conv_1 = nn.Sequential(
            nn.Conv3d(2, base_channel // 2, 3, 1, 1, bias=False),
            nn.Conv3d(
                base_channel // 2, base_channel // 2, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(base_channel // 2, eps=1e-6),
        )  # shape: [1,16,112,112]

        # Level 1: 1/2*origin_shape
        self.encoder_conv_2 = nn.Sequential(
            nn.Conv3d(base_channel // 2, base_channel, 3, 2, 1, bias=False),
            nn.Conv3d(base_channel, base_channel, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(base_channel, eps=1e-6),
        )  # shape: [1,16,56,56]

        # Level 2: 1/4*origin_shape
        self.patch_emb_1 = PatchEmbed(
            patch_size=patch_size,
            in_chans=base_channel,
            padding=self.pad,
            embed_dim=base_channel * int(2**1))  # [1,28*28,C]

        self.img_size = tuple(shape // down_ratio[0] for shape in img_size)

        self.encoder_vit_1 = VisionTransformer(
            img_size=self.img_size,
            patch_size=patch_size,
            embed_dim=base_channel * int(2**1),
            sr_ratio=sr_ratio[0],
            # depth=vit_depth,
            depth=4,
            n_heads=n_heads[0])  # [1,28*28,1*C]

        # Level 3: 1/8*origin_shape
        self.patch_emb_2 = nn.Sequential(
            Rearrange(
                'b (D H W) c -> b c D H W',
                D=self.img_size[0],
                H=self.img_size[1],
                W=self.img_size[2]),
            PatchEmbed(
                patch_size=patch_size,
                in_chans=base_channel * int(2**1),
                padding=self.pad,
                embed_dim=base_channel * int(2**2)),
        )  # [1,14*14,1*C]

        self.img_size = tuple(shape // down_ratio[1] for shape in img_size)

        self.encoder_vit_2 = VisionTransformer(
            img_size=self.img_size,
            patch_size=patch_size,
            embed_dim=base_channel * int(2**2),
            sr_ratio=sr_ratio[1],
            # depth=vit_depth,
            depth=2,
            n_heads=n_heads[1])  # [1,17*17,2*C]

        # Level 4: 1/16*origin_shape, bottle_neck
        self.patch_emb_3 = nn.Sequential(
            Rearrange(
                'b (D H W) c -> b c D H W',
                D=self.img_size[0],
                H=self.img_size[1],
                W=self.img_size[2]),
            PatchEmbed(
                patch_size=patch_size,
                in_chans=base_channel * int(2**2),
                padding=self.pad,
                embed_dim=base_channel * int(2**3)),
        )  # [1,7*7,2*C]

        self.img_size = tuple(shape // down_ratio[2] for shape in img_size)

        self.encoder_vit_3 = VisionTransformer(
            img_size=self.img_size,
            patch_size=patch_size,
            embed_dim=base_channel * int(2**3),
            sr_ratio=sr_ratio[2],
            # depth=vit_depth,
            depth=2,
            n_heads=n_heads[2])  # [1,7*7,4*C]
        """----------------------Decoder-----------------------"""

        self.patch_exp_1 = PatchExpanding(
            dim=base_channel * int(2**3),
            input_shape=self.encoder_vit_3.img_size)  # [1,12*14*12,2*C]
        # self.cat_fuse_1 = SkipConnection(self.encoder_vit_2.embed_dim)
        # output_shape:[1,2C,12,14,12]
        self.cat_fuse_1 = SkipConnection_v2(self.encoder_vit_2.embed_dim,
                                            self.encoder_vit_2.img_size)
        self.patch_emb_dec_1 = nn.Sequential(
            PatchEmbed(3, self.encoder_vit_2.embed_dim, 1,
                       self.encoder_vit_2.embed_dim,
                       1))  # output_shape:[1,12*14*12,2C]
        self.decoder_vit_1 = VisionTransformer(
            img_size=self.encoder_vit_2.img_size,
            patch_size=patch_size,
            embed_dim=self.encoder_vit_2.embed_dim,
            sr_ratio=sr_ratio[1],
            depth=2,
            n_heads=n_heads[1])  # [1,12*14*12,2*C]

        self.patch_exp_2 = PatchExpanding(
            dim=base_channel * int(2**2),
            input_shape=self.decoder_vit_1.img_size)  # [1,24*28*24,4*C]
        # self.cat_fuse_2 = SkipConnection(self.encoder_vit_1.embed_dim)
        # output_shape:[1,24,28,24,4C]
        self.cat_fuse_2 = SkipConnection_v2(self.encoder_vit_1.embed_dim,
                                            self.encoder_vit_1.img_size)
        self.patch_emb_dec_2 = nn.Sequential(
            PatchEmbed(3, self.encoder_vit_1.embed_dim, 1,
                       self.encoder_vit_1.embed_dim,
                       1))  # output_shape:[1,24*28*24,4C]
        self.decoder_vit_2 = VisionTransformer(
            img_size=self.encoder_vit_1.img_size,
            patch_size=patch_size,
            embed_dim=self.encoder_vit_1.embed_dim,
            sr_ratio=sr_ratio[0],
            depth=2,
            n_heads=n_heads[0])  # [1,24*28*24,4*C]
        self.decoder_conv_1 = nn.Sequential(
            Rearrange(
                'b (D H W) c -> b c D H W',
                D=self.encoder_vit_1.img_size[0],
                H=self.encoder_vit_1.img_size[1],
                W=self.encoder_vit_1.img_size[2],
            ))  # [1,32,24,28,24]

        self.upsample_1 = nn.ConvTranspose3d(
            in_channels=base_channel * 2,
            out_channels=base_channel * 1,
            kernel_size=2,
            stride=2,
            bias=True)
        self.decoder_conv_2 = nn.Sequential(
            nn.Conv3d(base_channel * 2, base_channel * 1, 3, 1, 1, bias=True),
            nn.Conv3d(base_channel * 1, base_channel * 1, 3, 1, 1, bias=True),
            nn.InstanceNorm3d(base_channel * 1, eps=1e-6))
        self.upsample_2 = nn.ConvTranspose3d(
            base_channel * 1, base_channel // 2, 2, 2, bias=True)
        self.decoder_conv_3 = nn.Conv3d(
            base_channel, base_channel // 2, 3, 1, 1, bias=True)

        # self.flow1 = nn.Sequential(
        #     nn.Conv3d(
        #         base_channel // 2, base_channel // 2, 3, 1, 1, bias=True),
        #     nn.ReLU(),
        #     nn.Conv3d(base_channel // 2, 3, 3, 1, 1, bias=True),
        # )

        # # diffeomorphic learning
        # if self.learning_mode == 'diffeomorphic':
        #     self.Vec = VecInt([96, 112, 96], 7)
        # else:
        #     pass

    def forward(self, source, target):

        x_conv1 = self.encoder_conv_1(torch.cat([source, target], 1))
        x_conv2 = self.encoder_conv_2(x_conv1)  # base_channel, size//2

        x = self.patch_emb_1(x_conv2)
        x_enc_vit1 = self.encoder_vit_1(x)

        x = self.patch_emb_2(x_enc_vit1)
        x_enc_vit2 = self.encoder_vit_2(x)

        x = self.patch_emb_3(x_enc_vit2)
        x_enc_vit3 = self.encoder_vit_3(x)
        x = self.patch_exp_1(x_enc_vit3)
        x = self.cat_fuse_1(x_enc_vit2, x)
        x = self.patch_emb_dec_1(x)
        x = self.decoder_vit_1(x)

        x = self.patch_exp_2(x)
        x = self.cat_fuse_2(x_enc_vit1, x)
        x = self.patch_emb_dec_2(x)
        x = self.decoder_vit_2(x)
        x = self.decoder_conv_1(x)  # Only reshape operation

        x = self.upsample_1(x)
        x = self.decoder_conv_2(torch.cat([x_conv2, x], 1))

        x = self.upsample_2(x)
        x = self.decoder_conv_3(torch.cat([x_conv1, x], 1))

        # flow = self.flow1(x)

        # if self.learning_mode == 'diffeomorphic':
        #     flow = self.Vec(flow)
        # else:
        #     pass

        # return flow
        return x


@MODELS.register_module()
class SymTrans(nn.Module):

    def __init__(
        self,
        img_size,
        n_heads,
        base_channel,
        down_ratio,
        patch_size,
        sr_ratio,
    ):
        super().__init__()

        self.RegTran = RegTran(
            img_size=img_size,
            base_channel=base_channel,
            down_ratio=down_ratio,
            # vit_depth=vit_depth,
            patch_size=patch_size,
            n_heads=n_heads,
            sr_ratio=sr_ratio,
        )

    def forward(self, source, target, **kwargs):
        pre_flow = self.RegTran(source, target)

        return pre_flow


@MODELS.register_module()
class SymTransFlow(nn.Module):

    def __init__(self,
                 img_size,
                 base_channel,
                 learning_mode='displacement') -> None:
        super().__init__()

        self.learning_mode = learning_mode
        self.flow1 = nn.Sequential(
            nn.Conv3d(
                base_channel // 2, base_channel // 2, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv3d(base_channel // 2, 3, 3, 1, 1, bias=True),
        )

        # diffeomorphic learning
        if self.learning_mode == 'diffeomorphic':
            self.Vec = VecInt(img_size, 7)
        else:
            pass

    def forward(self, x, **kwargs):
        flow = self.flow1(x)

        if self.learning_mode == 'diffeomorphic':
            flow = self.Vec(flow)
        else:
            pass

        return flow
