# Copyright (c) MMIPT. All rights reserved.
import numpy as np
import torch
import torch.nn as nn

from mmipt.registry import MODELS


class ConvBlock(nn.Module):
    """Specific convolutional block followed by leakyrelu for unet."""

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class Unet(nn.Module):
    """A unet architecture.

    Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 img_size,
                 nb_features=None,
                 nb_levels=None,
                 feat_mult=1):
        super().__init__()
        """
        Parameters:
            img_size: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(img_size)
        assert ndims in [
            1, 2, 3
        ], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = (
                (16, 32, 32, 32),  # encoder
                (32, 32, 32, 32, 32, 16, 16)  # decoder
            )

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError(
                    'must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features *
                             feat_mult**np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError(
                'cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

    def forward(self, x):
        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x


@MODELS.register_module()
class VxmDense(nn.Module):
    """VoxelMorph network for (unsupervised) nonlinear registration between two
    images."""

    def __init__(
        self,
        img_size,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False,
    ):
        """
        Parameters:
            img_size: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(img_size)
        assert ndims in [1, 2, 3], \
            'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            img_size,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
        )

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False'
            )

    def forward(self, source: torch.Tensor, target: torch.Tensor, **kwagrs):
        """
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        """

        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        return x


@MODELS.register_module()
class VxmDense1(VxmDense):

    def __init__(
        self,
        img_size,
        nb_unet_features=(
            (8, 32, 32, 32),
            (32, 32, 32, 32, 32, 8, 8),
        ),
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False,
    ):
        super().__init__(img_size, nb_unet_features, nb_unet_levels,
                         unet_feat_mult, use_probs)


@MODELS.register_module()
class VxmDense2(VxmDense):

    def __init__(
        self,
        img_size,
        nb_unet_features=(
            (16, 32, 32, 32),
            (32, 32, 32, 32, 32, 16, 16),
        ),
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False,
    ):
        super().__init__(img_size, nb_unet_features, nb_unet_levels,
                         unet_feat_mult, use_probs)


@MODELS.register_module()
class VxmDenseX2(nn.Module):

    def __init__(
        self,
        img_size,
        nb_unet_features=(
            (32, 64, 64, 64),
            (64, 64, 64, 64, 64, 32, 32),
        ),
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False,
    ):
        super().__init__(img_size, nb_unet_features, nb_unet_levels,
                         unet_feat_mult, use_probs)


@MODELS.register_module()
class VxmDenseHuge(nn.Module):

    def __init__(
        self,
        img_size,
        nb_unet_features=(
            (14, 28, 144, 320),
            (1152, 1152, 320, 144, 28, 14, 14),
        ),
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False,
    ):
        super().__init__(img_size, nb_unet_features, nb_unet_levels,
                         unet_feat_mult, use_probs)
