''' Define the sublayers in Deformer'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position):
        super().__init__()

        # Not a parameter
        self.register_buffer(
            'pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class LocalAttention(nn.Module):
    ''' Multi-Head local-Attention module '''

    def __init__(self, n_head, n_point, d_model):
        super().__init__()

        self.n_head = n_head
        self.n_point = n_point
        # one linear layer to obtain displacement basis
        self.sampling_offsets = nn.Linear(d_model, n_head * n_point * 3)
        # one linear layer to obtain weight
        self.attention_weights = nn.Linear(2 * d_model, n_head * n_point)
        # self.attention_weights = nn.Linear(2 * d_model, n_head * 3)

    def forward(self, q, k):
        v = torch.cat([q, k], dim=-1)
        n_head, n_point = self.n_head, self.n_point
        sz_b, len_q, len_k = q.size(0), q.size(1), k.size(1)
        # left branch (only moving image)
        sampling_offsets = self.sampling_offsets(q).view(
            sz_b, len_q, n_head, n_point, 3)
        # right branch (concat moving and fixed image)
        attn = self.attention_weights(v).view(sz_b, len_q, n_head, n_point, 1)
        # attn = self.attention_weights(v).view(sz_b, len_q, n_head, 3)
        # flow = attn
        # attn = F.softmax(attn, dim=-2)
        # multiple and head-wise average
        flow = torch.matmul(sampling_offsets.transpose(3, 4), attn)
        flow = torch.squeeze(flow, dim=-1)
        # sz_b, len_q, 3
        return torch.mean(flow, dim=-2)


class DeformerLayer(nn.Module):
    ''' Compose layers '''

    def __init__(self, d_model, n_head, n_point):
        super().__init__()
        self.slf_attn = LocalAttention(n_head, n_point, d_model)

    def forward(self, enc_input, enc_input1):
        enc_output = self.slf_attn(enc_input, enc_input1)
        return enc_output


class Deformer(nn.Module):
    '''
    A encoder model with deformer mechanism.
    :param n_layers: the number of layer.
    :param d_model: the channel of input image [batch,N,d_model].
    :param n_position: input image [batch,N,d_model], n_position=N.
    :param n_head: the number of head.
    :param n_point: the number of displacement base.
    :param src_seq: moving seq [batch,N,d_model]
    :param tgt_seq: fixed seq [batch,N,d_model].
    :return enc_output: sub flow field [batch,N,3].
    '''

    def __init__(self,
                 n_layers,
                 d_model,
                 n_position,
                 n_head,
                 n_point,
                 dropout=0.1,
                 scale_emb=False):

        super().__init__()

        self.position_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DeformerLayer(d_model, n_head, n_point) for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, tgt_seq):

        # -- Forward
        if self.scale_emb:
            src_seq *= self.d_model**0.5
            tgt_seq *= self.d_model**0.5
        enc_output = self.dropout(self.position_enc(src_seq))
        enc_output = self.layer_norm(enc_output)
        enc_output1 = self.dropout(self.position_enc(tgt_seq))
        enc_output1 = self.layer_norm(enc_output1)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, enc_output1)

        return enc_output


class DeformableSkipLearner(nn.Module):
    '''The Refining Network of DMR'''

    def __init__(self, inch):
        super().__init__()

        def make_building_block(in_channel,
                                out_channels,
                                kernel_sizes,
                                spt_strides,
                                group=1):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(
                    zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                pad = ksz // 2
                building_block_layers.append(
                    nn.Conv3d(inch, outch, ksz, stride, pad))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3 = 16, 64, 128

        # Squeezing building blocks
        self.encoder_layer4 = make_building_block(inch[0],
                                                  [outch1, outch2, outch3],
                                                  [3, 3, 3], [1, 1, 1])
        self.encoder_layer3 = make_building_block(inch[1],
                                                  [outch1, outch2, outch3],
                                                  [5, 3, 3], [1, 1, 1])
        self.encoder_layer2 = make_building_block(inch[2],
                                                  [outch1, outch2, outch3],
                                                  [5, 5, 3], [1, 1, 1])
        self.encoder_layer1 = make_building_block(inch[3],
                                                  [outch1, outch2, outch3],
                                                  [5, 5, 5], [1, 1, 1])

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(outch3 + 32 * 2,
                                                     [outch3, outch3, outch3],
                                                     [3, 3, 3], [1, 1, 1])
        self.encoder_layer3to2 = make_building_block(outch3 + 32 * 2,
                                                     [outch3, outch3, outch3],
                                                     [3, 3, 3], [1, 1, 1])
        self.encoder_layer2to1 = make_building_block(outch3 + 16 * 2,
                                                     [outch3, outch3, outch3],
                                                     [3, 3, 3], [1, 1, 1])

        # Decoder layers
        self.decoder1 = nn.Sequential(
            nn.Conv3d(outch3, outch3, (3, 3, 3), padding=(1, 1, 1), bias=True),
            nn.ReLU(),
            nn.Conv3d(outch3, outch2, (3, 3, 3), padding=(1, 1, 1), bias=True),
            nn.ReLU())

        self.decoder2 = nn.Sequential(
            nn.Conv3d(outch2, outch2, (3, 3, 3), padding=(1, 1, 1), bias=True),
            nn.ReLU(),
            nn.Conv3d(outch2, outch1, (3, 3, 3), padding=(1, 1, 1), bias=True),
            nn.ReLU())

        self.decoder3 = nn.Sequential(
            nn.Conv3d(outch1, outch1, (3, 3, 3), padding=(1, 1, 1), bias=True),
            nn.ReLU(),
            nn.Conv3d(outch1, 3, (3, 3, 3), padding=(1, 1, 1), bias=True))

    def interpolate_dims(self, hypercorr, spatial_size=None):
        bsz, ch, d, w, h = hypercorr.size()
        hypercorr = F.interpolate(
            hypercorr, (2 * d, 2 * w, 2 * h),
            mode='trilinear',
            align_corners=True)
        return hypercorr

    def forward(self, hypercorr_pyramid, moving_feat, fixed_feat):
        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])
        hypercorr_sqz1 = self.encoder_layer1(hypercorr_pyramid[3])

        # Propagate encoded 3D-tensor (Mixing building blocks)
        hypercorr_sqz4 = self.interpolate_dims(hypercorr_sqz4,
                                               hypercorr_sqz3.size()[-6:-3])
        hypercorr_mix43 = 2 * hypercorr_sqz4 + hypercorr_sqz3  #add
        hypercorr_mix43 = torch.cat(
            [hypercorr_mix43, moving_feat[-2], fixed_feat[-2]],
            dim=1)  #skip connection
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        hypercorr_mix43 = self.interpolate_dims(hypercorr_mix43,
                                                hypercorr_sqz2.size()[-6:-3])
        hypercorr_mix432 = 2 * hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = torch.cat(
            [hypercorr_mix432, moving_feat[-3], fixed_feat[-3]], dim=1)
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        hypercorr_mix432 = self.interpolate_dims(hypercorr_mix432,
                                                 hypercorr_sqz1.size()[-6:-3])
        hypercorr_mix4321 = 2 * hypercorr_mix432 + hypercorr_sqz1
        hypercorr_mix4321 = torch.cat(
            [hypercorr_mix4321, moving_feat[-4], fixed_feat[-4]], dim=1)
        hypercorr_mix4321 = self.encoder_layer2to1(hypercorr_mix4321)

        # Decode the encoded 3D-tensor
        hypercorr_decoded = self.decoder1(hypercorr_mix4321)
        upsample_size = (hypercorr_decoded.size(-3) * 2,
                         hypercorr_decoded.size(-2) * 2,
                         hypercorr_decoded.size(-1) * 2)
        hypercorr_decoded = 2 * F.interpolate(
            hypercorr_decoded,
            upsample_size,
            mode='trilinear',
            align_corners=True)
        hypercorr_decoded = self.decoder2(hypercorr_decoded)
        logit_mask = self.decoder3(hypercorr_decoded)

        return logit_mask


class SpatialTransformer(nn.Module):
    '''Refer to STN (paper "Spatial Transformer Networks")'''

    def __init__(self, size, mode='bilinear'):
        super().__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (
                new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(
            src, new_locs, mode=self.mode, align_corners=False)


if __name__ == '__main__':
    # module 1
    x = torch.rand(3, 112 * 96 * 80, 16)
    y = torch.rand(3, 112 * 96 * 80, 16)
    b, n, d = x.size()
    enc = Deformer(n_layers=1, d_model=d, n_position=n, n_head=8, n_point=64)
    z = enc(x, y)
    print(z.size())
    print(torch.min(z))

    # module 2
    corr = []
    corr.append(torch.rand(2, 3, 14, 12, 10))
    corr.append(torch.rand(2, 3, 28, 24, 20))
    corr.append(torch.rand(2, 3, 56, 48, 40))
    corr.append(torch.rand(2, 3, 112, 96, 80))
    hpn_learner = Deformable_Skip_Learner([3, 3, 3, 3])
    moving_feat = []
    moving_feat.append(torch.rand(2, 16, 112, 96, 80))
    moving_feat.append(torch.rand(2, 32, 56, 48, 40))
    moving_feat.append(torch.rand(2, 32, 28, 24, 20))
    moving_feat.append(torch.rand(2, 64, 14, 12, 10))

    fixed_feat = []
    fixed_feat.append(torch.rand(2, 16, 112, 96, 80))
    fixed_feat.append(torch.rand(2, 32, 56, 48, 40))
    fixed_feat.append(torch.rand(2, 32, 28, 24, 20))
    fixed_feat.append(torch.rand(2, 64, 14, 12, 10))

    y = hpn_learner(corr, moving_feat, fixed_feat)
    print(y.shape)
