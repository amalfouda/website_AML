"""
Standalone RECCE model — no DeepfakeBench registry dependency.
Classes copied verbatim from DeepfakeBench/training/detectors/recce_detector.py
and DeepfakeBench/training/networks/xception.py, keeping only what is needed
for inference.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import xception


# ── Helpers copied from networks/xception.py ───────────────────────────────

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation,
                               groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.conv1(x))


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1,
                 start_with_relu=True, grow_first=True):
        super().__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters

        if grow_first:
            rep.append(relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for _ in range(reps - 1):
            rep.append(relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        return x + skip


# ── RECCE components ────────────────────────────────────────────────────────

class GraphReasoning(nn.Module):
    """Graph Reasoning Module for information aggregation."""

    def __init__(self, va_in, va_out, vb_in, vb_out, vc_in, vc_out,
                 spatial_ratio, drop_rate):
        super().__init__()
        self.ratio = spatial_ratio
        self.va_embedding = nn.Sequential(
            nn.Conv2d(va_in, va_out, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(va_out, va_out, 1, bias=False),
        )
        self.va_gated_b = nn.Sequential(
            nn.Conv2d(va_in, va_out, 1, bias=False), nn.Sigmoid())
        self.va_gated_c = nn.Sequential(
            nn.Conv2d(va_in, va_out, 1, bias=False), nn.Sigmoid())
        self.vb_embedding = nn.Sequential(
            nn.Linear(vb_in, vb_out, bias=False), nn.ReLU(True),
            nn.Linear(vb_out, vb_out, bias=False))
        self.vc_embedding = nn.Sequential(
            nn.Linear(vc_in, vc_out, bias=False), nn.ReLU(True),
            nn.Linear(vc_out, vc_out, bias=False))
        self.unfold_b = nn.Unfold(kernel_size=spatial_ratio[0], stride=spatial_ratio[0])
        self.unfold_c = nn.Unfold(kernel_size=spatial_ratio[1], stride=spatial_ratio[1])
        self.reweight_ab = nn.Sequential(
            nn.Linear(va_out + vb_out, 1, bias=False), nn.ReLU(True), nn.Softmax(dim=1))
        self.reweight_ac = nn.Sequential(
            nn.Linear(va_out + vc_out, 1, bias=False), nn.ReLU(True), nn.Softmax(dim=1))
        self.reproject = nn.Sequential(
            nn.Conv2d(va_out + vb_out + vc_out, va_in, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(va_in, va_in, kernel_size=1, bias=False),
            nn.Dropout(drop_rate) if drop_rate else nn.Identity(),
        )

    def forward(self, vert_a, vert_b, vert_c):
        emb_a = self.va_embedding(vert_a)
        emb_a = emb_a.reshape([emb_a.shape[0], emb_a.shape[1], -1])

        gate_b = (1 - self.va_gated_b(vert_a)).reshape(*emb_a.shape)
        gate_c = (1 - self.va_gated_c(vert_a)).reshape(*emb_a.shape)

        vert_b = self.unfold_b(vert_b).reshape(
            [vert_b.shape[0], vert_b.shape[1], self.ratio[0] ** 2, -1]).permute([0, 2, 3, 1])
        emb_b = self.vb_embedding(vert_b)

        vert_c = self.unfold_c(vert_c).reshape(
            [vert_c.shape[0], vert_c.shape[1], self.ratio[1] ** 2, -1]).permute([0, 2, 3, 1])
        emb_c = self.vc_embedding(vert_c)

        agg_b, agg_c = [], []
        for j in range(emb_a.shape[-1]):
            ev_a = torch.stack([emb_a[:, :, j]] * (self.ratio[0] ** 2), dim=1)
            ev_b = emb_b[:, :, j, :]
            w = self.reweight_ab(torch.cat([ev_a, ev_b], dim=-1))
            agg_b.append(torch.bmm(ev_b.transpose(1, 2), w).squeeze() * gate_b[:, :, j])

            ev_a = torch.stack([emb_a[:, :, j]] * (self.ratio[1] ** 2), dim=1)
            ev_c = emb_c[:, :, j, :]
            w = self.reweight_ac(torch.cat([ev_a, ev_c], dim=-1))
            agg_c.append(torch.bmm(ev_c.transpose(1, 2), w).squeeze() * gate_c[:, :, j])

        agg_bc = torch.cat([torch.stack(agg_b, -1), torch.stack(agg_c, -1)], dim=1)
        agg_abc = torch.sigmoid(torch.cat([agg_bc, emb_a], dim=1))
        agg_abc = agg_abc.reshape(vert_a.shape[0], -1, vert_a.shape[2], vert_a.shape[3])
        return self.reproject(agg_abc)


class GuidedAttention(nn.Module):
    """Reconstruction Guided Attention."""

    def __init__(self, depth=728, drop_rate=0.2):
        super().__init__()
        self.depth = depth
        self.gated = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(3, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.h = nn.Sequential(
            nn.Conv2d(depth, depth, 1, 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.ReLU(True),
        )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, pred_x, embedding):
        residual_full = torch.abs(x - pred_x)
        residual_x = F.interpolate(residual_full, size=embedding.shape[-2:],
                                   mode='bilinear', align_corners=True)
        res_map = self.gated(residual_x)
        return res_map * self.h(embedding) + self.dropout(embedding)


class CBAM(nn.Module):
    """Convolutional Block Attention Module with identity initialization."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True),
            nn.Sigmoid(),
        )
        # Near-identity init — preserves pretrained XCeption features at epoch 0
        nn.init.zeros_(self.channel_attention[-2].weight)
        nn.init.constant_(self.channel_attention[-2].bias, 3.0)
        nn.init.zeros_(self.spatial_attention[0].weight)
        nn.init.constant_(self.spatial_attention[0].bias, 3.0)

    def forward(self, x):
        ca = self.channel_attention(x).unsqueeze(2).unsqueeze(3)
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        return x * sa


class FrequencyBranch(nn.Module):
    """FFT-based frequency branch (disabled at inference — kept for checkpoint compatibility)."""

    def __init__(self, in_channels=3, features=2048):
        super().__init__()
        self.dct_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, features)
        self.gate = nn.Sequential(nn.Linear(features * 2, features), nn.Sigmoid())
        nn.init.zeros_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, -3.0)

    def forward(self, x, spatial_features):
        x_freq = torch.log1p(torch.abs(torch.fft.fft2(x.float(), norm='ortho'))).float()
        freq_feat = self.fc(self.pool(self.dct_conv(x_freq)).squeeze(2).squeeze(2))
        gate_w = self.gate(torch.cat([spatial_features, freq_feat], dim=1))
        return spatial_features + gate_w * freq_feat


# ── Main model ──────────────────────────────────────────────────────────────

_ENCODER_FEATURES = 2048

class Recce(nn.Module):
    """End-to-End Reconstruction-Classification Learning for Face Forgery Detection."""

    # Manipulation-type class names (for the 4-class head)
    MANIPULATION_CLASSES = ['DeepFakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

    def __init__(self, num_classes=2, drop_rate=0.2, num_types=4):
        super().__init__()
        self.name = "xception"
        self.loss_inputs = dict()
        self.encoder = xception(pretrained=False)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(_ENCODER_FEATURES, num_classes)

        # ── Multi-task heads ───────────────────────────────────────────────
        # Head 1: manipulation-type classifier (DF / F2F / FS / NT)
        self.type_head = nn.Sequential(
            nn.Linear(_ENCODER_FEATURES, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_types),
        )
        # Head 2: binary forgery-region mask predictor
        self.mask_project = nn.Linear(_ENCODER_FEATURES, 512 * 4 * 4)
        self.mask_decode = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),          # 8 x 8
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),          # 16 x 16
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),          # 32 x 32
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),          # 64 x 64
            nn.Conv2d(64, 1, 1, bias=True),
            nn.Sigmoid(),
        )

        self.attention = GuidedAttention(depth=728, drop_rate=drop_rate)
        # FrequencyBranch is kept for checkpoint key compatibility but not called
        # self.freq_branch = FrequencyBranch(in_channels=3, features=_ENCODER_FEATURES)
        self.cbam1 = CBAM(channels=128)
        self.cbam2 = CBAM(channels=256)
        self.cbam3 = CBAM(channels=728)
        self.reasoning = GraphReasoning(728, 256, 256, 256, 128, 256, [2, 4], drop_rate)

        self.decoder1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(728, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.decoder2 = Block(256, 256, 3, 1)
        self.decoder3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.decoder4 = Block(128, 128, 3, 1)
        self.decoder5 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.decoder6 = nn.Sequential(
            nn.Conv2d(64, 3, 1, 1, bias=False), nn.Tanh())

    def norm_n_corr(self, x):
        norm_embed = F.normalize(self.global_pool(x), p=2, dim=1)
        corr = (torch.matmul(norm_embed.squeeze(), norm_embed.squeeze().T) + 1.0) / 2.0
        return norm_embed, corr

    @staticmethod
    def add_white_noise(tensor, mean=0., std=1e-6):
        rand = torch.where(torch.rand([tensor.shape[0], 1, 1, 1]) > 0.5,
                           torch.ones(1), torch.zeros(1)).to(tensor.device)
        noise = torch.normal(mean, std, size=tensor.shape, device=tensor.device)
        return torch.clip(tensor + noise * rand, -1., 1.)

    def features(self, x):
        self.loss_inputs = dict(recons=[], contra=[])
        noise_x = self.add_white_noise(x) if self.training else x

        out = self.encoder.conv1(noise_x)
        out = self.encoder.bn1(out)
        out = self.encoder.act1(out)
        out = self.encoder.conv2(out)
        out = self.encoder.bn2(out)
        out = self.encoder.act2(out)

        out = self.cbam1(self.encoder.block1(out))
        out = self.cbam2(self.encoder.block2(out))
        out = self.encoder.block3(out)
        embedding = self.cbam3(self.encoder.block4(out))

        _, corr = self.norm_n_corr(embedding)
        self.loss_inputs['contra'].append(corr)

        out = self.decoder1(self.dropout(embedding))
        out_d2 = self.decoder2(out)
        _, corr = self.norm_n_corr(out_d2)
        self.loss_inputs['contra'].append(corr)

        out = self.decoder3(out_d2)
        out_d4 = self.decoder4(out)
        _, corr = self.norm_n_corr(out_d4)
        self.loss_inputs['contra'].append(corr)

        out = self.decoder5(out_d4)
        pred = self.decoder6(out)
        recons_x = F.interpolate(pred, size=x.shape[-2:], mode='bilinear', align_corners=True)
        self.loss_inputs['recons'].append(recons_x)

        embedding = self.encoder.block5(embedding)
        embedding = self.encoder.block6(embedding)
        embedding = self.encoder.block7(embedding)

        fusion = self.reasoning(embedding, out_d2, out_d4) + embedding
        embedding = self.encoder.block8(fusion)
        img_att = self.attention(x, recons_x, embedding)

        embedding = self.encoder.block9(img_att)
        embedding = self.encoder.block10(embedding)
        embedding = self.encoder.block11(embedding)
        embedding = self.encoder.block12(embedding)

        embedding = self.encoder.conv3(embedding)
        embedding = self.encoder.bn3(embedding)
        embedding = self.encoder.act3(embedding)
        embedding = self.encoder.conv4(embedding)
        embedding = self.encoder.bn4(embedding)
        embedding = self.encoder.act4(embedding)

        embedding = self.global_pool(embedding).squeeze(2).squeeze(2)
        embedding = self.dropout(embedding)
        # freq_branch disabled — gate learned to close on C23-only training
        return embedding

    def classifier(self, embedding):
        return self.fc(embedding)

    def forward(self, x):
        return self.classifier(self.features(x))

    def forward_multitask(self, x):
        """Multi-task forward pass.

        Returns a dict:
            fake_logits  – (B, 2)            real/fake binary logits
            type_logits  – (B, num_types)    manipulation-type logits (DF/F2F/FS/NT)
            mask         – (B, 1, 64, 64)    forgery-region probability map in [0, 1]
        """
        embedding = self.features(x)
        mask_spatial = self.mask_project(embedding).reshape(-1, 512, 4, 4)
        return {
            'fake_logits': self.fc(embedding),
            'type_logits': self.type_head(embedding),
            'mask':        self.mask_decode(mask_spatial),
        }
