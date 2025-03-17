# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# VQGAN implementation using Hugging Face Trainer

import math
import argparse
import numpy as np
import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    HfArgumentParser,
    TrainerCallback
)
from dataclasses import dataclass, field


# Utility functions

def shift_dim(x, src_dim, dest_dim):
    """Shifts a dimension from src_dim to dest_dim"""
    dims = list(range(len(x.shape)))
    dims.pop(src_dim)
    dims.insert(dest_dim, src_dim)
    return x.permute(*dims)


def adopt_weight(step, threshold=0):
    """Gradually increase a weight from 0 to 1 after the threshold step"""
    if threshold == 0:
        return 1.0
    return 1.0 if step >= threshold else 0.0


def comp_getattr(obj, attr):
    """Get attribute via dot notation"""
    for a in attr.split('.'):
        obj = getattr(obj, a)
    return obj


def silu(x):
    """SiLU activation function"""
    return x * torch.sigmoid(x)


# Model components

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)


def hinge_d_loss(logits_real, logits_fake):
    """Hinge loss for discriminator"""
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    """Vanilla GAN loss for discriminator"""
    d_loss = 0.5 * (
            torch.mean(torch.nn.functional.softplus(-logits_real)) +
            torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class Normalize(nn.Module):
    def __init__(self, in_channels, norm_type='group'):
        super().__init__()
        assert norm_type in ['group', 'batch']
        if norm_type == 'group':
            # Ensure num_groups doesn't exceed num_channels
            num_groups = min(32, in_channels)
            self.norm = torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        elif norm_type == 'batch':
            self.norm = torch.nn.BatchNorm3d(in_channels)

    def forward(self, x):
        return self.norm(x)


class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode=self.padding_type))


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input, mode=self.padding_type))


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group',
                 padding_type='replicate'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = SamePadConv3d(in_channels, out_channels, kernel_size=3, padding_type=padding_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = Normalize(out_channels, norm_type)
        self.conv2 = SamePadConv3d(out_channels, out_channels, kernel_size=3, padding_type=padding_type)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3d(in_channels, out_channels, kernel_size=3, padding_type=padding_type)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x + h


class Encoder(nn.Module):
    def __init__(self, n_hiddens, downsample, image_channel=3, norm_type='group', padding_type='replicate'):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.conv_blocks = nn.ModuleList()
        max_ds = n_times_downsample.max()

        self.conv_first = SamePadConv3d(image_channel, n_hiddens, kernel_size=3, padding_type=padding_type)

        for i in range(max_ds):
            block = nn.Module()
            in_channels = n_hiddens * 2 ** i
            out_channels = n_hiddens * 2 ** (i + 1)
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            block.down = SamePadConv3d(in_channels, out_channels, 4, stride=stride, padding_type=padding_type)
            block.res = ResBlock(out_channels, out_channels, norm_type=norm_type)
            self.conv_blocks.append(block)
            n_times_downsample -= 1

        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type),
            SiLU()
        )

        self.out_channels = out_channels

    def forward(self, x):
        h = self.conv_first(x)
        for block in self.conv_blocks:
            h = block.down(h)
            h = block.res(h)
        h = self.final_block(h)
        return h


class Decoder(nn.Module):
    def __init__(self, n_hiddens, upsample, image_channel, norm_type='group'):
        super().__init__()

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()

        in_channels = n_hiddens * 2 ** max_us
        self.final_block = nn.Sequential(
            Normalize(in_channels, norm_type),
            SiLU()
        )

        self.conv_blocks = nn.ModuleList()
        for i in range(max_us):
            block = nn.Module()
            in_channels = in_channels if i == 0 else n_hiddens * 2 ** (max_us - i + 1)
            out_channels = n_hiddens * 2 ** (max_us - i)
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            block.up = SamePadConvTranspose3d(in_channels, out_channels, 4, stride=us)
            block.res1 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            block.res2 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            self.conv_blocks.append(block)
            n_times_upsample -= 1

        self.conv_last = SamePadConv3d(out_channels, image_channel, kernel_size=3)

    def forward(self, x):
        h = self.final_block(x)
        for i, block in enumerate(self.conv_blocks):
            h = block.up(h)
            h = block.res1(h)
            h = block.res2(h)
        h = self.conv_last(h)
        return h


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), None


class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), None


class LPIPS(nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        # This is a simplified version - in a real implementation you would
        # include the VGG model with pre-trained weights
        self.dummy_layer = nn.Conv2d(3, 3, 1)  # Placeholder

    def forward(self, x, y):
        # Simplified implementation - returns mean squared error
        return F.mse_loss(x, y)


class Codebook(nn.Module):
    def __init__(self, n_codes, embedding_dim, no_random_restart=False, restart_thres=1.0):
        super().__init__()
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        # z: [b, c, t, h, w]
        self._need_init = False
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        y = self._tile(flat_inputs)

        d = y.shape[0]
        _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))

    def forward(self, z):
        # z: [b, c, t, h, w]
        if self._need_init and self.training:
            self._init_embeddings(z)
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)  # [bthw, c]

        # Ensure embeddings has the same device as flat_inputs
        embeddings = self.embeddings.to(flat_inputs.device)

        # Calculate distances
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * flat_inputs @ embeddings.t() \
                    + (embeddings.t() ** 2).sum(dim=0, keepdim=True)  # [bthw, n_codes]

        encoding_indices = torch.argmin(distances, dim=1)
        encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(flat_inputs)  # [bthw, ncode]
        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:])  # [b, t, h, w]

        embeddings = F.embedding(encoding_indices, self.embeddings.to(z.device))  # [b, t, h, w, c]
        embeddings = shift_dim(embeddings, -1, 1)  # [b, c, t, h, w]

        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

        # EMA codebook update
        if self.training:
            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            if not self.no_random_restart:
                usage = (self.N.view(self.n_codes, 1) >= self.restart_thres).float()
                self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        embeddings_st = (embeddings - z).detach() + z

        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return dict(embeddings=embeddings_st, encodings=encoding_indices,
                    commitment_loss=commitment_loss, perplexity=perplexity)

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings


# Main VQGAN Model
class VQGAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding_dim = args.embedding_dim
        self.n_codes = args.n_codes
        self.global_step = 0

        if not hasattr(args, 'padding_type'):
            args.padding_type = 'replicate'
        self.encoder = Encoder(args.n_hiddens, args.downsample, args.image_channels, args.norm_type, args.padding_type)
        self.decoder = Decoder(args.n_hiddens, args.downsample, args.image_channels, args.norm_type)
        self.enc_out_ch = self.encoder.out_channels
        self.pre_vq_conv = SamePadConv3d(self.enc_out_ch, args.embedding_dim, 1, padding_type=args.padding_type)
        self.post_vq_conv = SamePadConv3d(args.embedding_dim, self.enc_out_ch, 1)

        self.codebook = Codebook(args.n_codes, args.embedding_dim, no_random_restart=args.no_random_restart,
                                 restart_thres=args.restart_thres)

        self.gan_feat_weight = args.gan_feat_weight
        self.image_discriminator = NLayerDiscriminator(args.image_channels, args.disc_channels, args.disc_layers)
        self.video_discriminator = NLayerDiscriminator3D(args.image_channels, args.disc_channels, args.disc_layers)

        if args.disc_loss_type == 'vanilla':
            self.disc_loss = vanilla_d_loss
        elif args.disc_loss_type == 'hinge':
            self.disc_loss = hinge_d_loss

        self.perceptual_model = LPIPS().eval()

        self.image_gan_weight = args.image_gan_weight
        self.video_gan_weight = args.video_gan_weight

        self.perceptual_weight = args.perceptual_weight

        self.l1_weight = args.l1_weight

    @property
    def latent_shape(self):
        input_shape = (self.args.sequence_length // self.args.sample_every_n_frames, self.args.resolution,
                       self.args.resolution)
        return tuple([s // d for s, d in zip(input_shape,
                                             self.args.downsample)])

    def encode(self, x, include_embeddings=False):
        h = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(h)
        if include_embeddings:
            return vq_output['embeddings'], vq_output['encodings']
        else:
            return vq_output['encodings']

    def decode(self, encodings):
        h = F.embedding(encodings, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h)

    def forward(self, x, optimizer_idx=None, log_image=False):
        B, C, T, H, W = x.shape

        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']))

        recon_loss = F.l1_loss(x_recon, x) * self.l1_weight

        # Make sure frame_idx is within range
        max_t = max(1, T - 1)  # Ensure T is at least 1
        frame_idx = torch.randint(0, max_t, [B]).to(x.device)
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
        frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

        if log_image:
            return frames, frames_recon, x, x_recon

        if optimizer_idx == 0:
            # autoencoder
            perceptual_loss = 0
            if self.perceptual_weight > 0:
                perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight

            logits_image_fake, pred_image_fake = self.image_discriminator(frames_recon)
            logits_video_fake, pred_video_fake = self.video_discriminator(x_recon)
            g_image_loss = -torch.mean(logits_image_fake)
            g_video_loss = -torch.mean(logits_video_fake)
            g_loss = self.image_gan_weight * g_image_loss + self.video_gan_weight * g_video_loss

            disc_factor = adopt_weight(self.global_step, threshold=self.args.discriminator_iter_start)
            aeloss = disc_factor * g_loss

            # gan feature matching loss
            image_gan_feat_loss = 0
            video_gan_feat_loss = 0
            feat_weights = 4.0 / (3 + 1)
            if self.image_gan_weight > 0:
                logits_image_real, pred_image_real = self.image_discriminator(frames)
                for i in range(len(pred_image_fake) - 1):
                    image_gan_feat_loss += feat_weights * F.l1_loss(pred_image_fake[i], pred_image_real[i].detach()) * (
                                self.image_gan_weight > 0)
            if self.video_gan_weight > 0:
                logits_video_real, pred_video_real = self.video_discriminator(x)
                for i in range(len(pred_video_fake) - 1):
                    video_gan_feat_loss += feat_weights * F.l1_loss(pred_video_fake[i], pred_video_real[i].detach()) * (
                                self.video_gan_weight > 0)

            gan_feat_loss = disc_factor * self.gan_feat_weight * (image_gan_feat_loss + video_gan_feat_loss)

            return {
                'loss': recon_loss + vq_output['commitment_loss'] + aeloss + perceptual_loss + gan_feat_loss,
                'recon_loss': recon_loss,
                'commitment_loss': vq_output['commitment_loss'],
                'perceptual_loss': perceptual_loss,
                'aeloss': aeloss,
                'gan_feat_loss': gan_feat_loss,
                'g_image_loss': g_image_loss,
                'g_video_loss': g_video_loss,
                'image_gan_feat_loss': image_gan_feat_loss,
                'video_gan_feat_loss': video_gan_feat_loss,
                'perplexity': vq_output['perplexity']
            }

        if optimizer_idx == 1:
            # discriminator
            logits_image_real, _ = self.image_discriminator(frames.detach())
            logits_video_real, _ = self.video_discriminator(x.detach())

            logits_image_fake, _ = self.image_discriminator(frames_recon.detach())
            logits_video_fake, _ = self.video_discriminator(x_recon.detach())

            d_image_loss = self.disc_loss(logits_image_real, logits_image_fake)
            d_video_loss = self.disc_loss(logits_video_real, logits_video_fake)
            disc_factor = adopt_weight(self.global_step, threshold=self.args.discriminator_iter_start)
            discloss = disc_factor * (self.image_gan_weight * d_image_loss + self.video_gan_weight * d_video_loss)

            return {
                'loss': discloss,
                'd_image_loss': d_image_loss,
                'd_video_loss': d_video_loss,
                'logits_image_real': logits_image_real.mean(),
                'logits_image_fake': logits_image_fake.mean(),
                'logits_video_real': logits_video_real.mean(),
                'logits_video_fake': logits_video_fake.mean(),
            }

        perceptual_loss = self.perceptual_model(frames, frames_recon) * self.perceptual_weight
        return {
            'loss': recon_loss + vq_output['commitment_loss'] + perceptual_loss,
            'recon_loss': recon_loss,
            'commitment_loss': vq_output['commitment_loss'],
            'perceptual_loss': perceptual_loss,
            'perplexity': vq_output['perplexity']
        }


# VideoDataset with dummy data
class DummyVideoDataset(Dataset):
    def __init__(self, sequence_length=16, resolution=64, num_samples=100):
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create random video data
        video = torch.rand(3, self.sequence_length, self.resolution, self.resolution)
        return {'video': video}


# Arguments for HF Trainer
@dataclass
class VQGANArguments:
    embedding_dim: int = field(default=32, metadata={"help": "Dimension of embeddings"})
    n_codes: int = field(default=512, metadata={"help": "Number of codes in codebook"})
    n_hiddens: int = field(default=64, metadata={"help": "Number of hidden units"})
    downsample: List[int] = field(default_factory=lambda: [2, 2, 2], metadata={"help": "Downsampling factors"})
    image_channels: int = field(default=3, metadata={"help": "Number of image channels"})
    disc_channels: int = field(default=64, metadata={"help": "Number of discriminator channels"})
    disc_layers: int = field(default=3, metadata={"help": "Number of discriminator layers"})
    discriminator_iter_start: int = field(default=5000, metadata={"help": "Step to start using discriminator"})
    disc_loss_type: str = field(default="hinge", metadata={"help": "Discriminator loss type"})
    image_gan_weight: float = field(default=1.0, metadata={"help": "Weight for image GAN loss"})
    video_gan_weight: float = field(default=1.0, metadata={"help": "Weight for video GAN loss"})
    l1_weight: float = field(default=4.0, metadata={"help": "Weight for L1 loss"})
    gan_feat_weight: float = field(default=0.0, metadata={"help": "Weight for GAN feature matching loss"})
    perceptual_weight: float = field(default=0.0, metadata={"help": "Weight for perceptual loss"})
    no_random_restart: bool = field(default=False, metadata={"help": "Disable random restart for unused codes"})
    restart_thres: float = field(default=1.0, metadata={"help": "Threshold for random restart"})
    norm_type: str = field(default="group", metadata={"help": "Normalization type"})
    padding_type: str = field(default="replicate", metadata={"help": "Padding type"})
    sequence_length: int = field(default=8, metadata={"help": "Sequence length"})
    resolution: int = field(default=32, metadata={"help": "Image resolution"})
    sample_every_n_frames: int = field(default=1, metadata={"help": "Sample every n frames"})
    data_path: str = field(default="./data", metadata={"help": "Path to data"})
    use_dummy_data: bool = field(default=True, metadata={"help": "Use dummy data instead of real data"})
    dummy_data_samples: int = field(default=100, metadata={"help": "Number of dummy data samples"})


# Custom VQGANTrainer that extends HF Trainer
class VQGANTrainer(Trainer):
    def __init__(self, vqgan_args=None, **kwargs):
        super().__init__(**kwargs)
        self.vqgan_args = vqgan_args

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Override for custom behavior.
        """
        # Extract video from inputs
        x = inputs['video'].to(self.args.device)

        # Update global_step in model
        model.global_step = self.state.global_step

        optimizer_idx = 0 if self.state.global_step % 2 == 0 else 1

        # Forward pass
        outputs = model(x, optimizer_idx=optimizer_idx)
        loss = outputs['loss']

        # Log metrics
        self._log_loss_metrics(outputs)

        return (loss, outputs) if return_outputs else loss

    def _log_loss_metrics(self, outputs):
        """Log all the loss components"""
        metrics = {}
        for key, value in outputs.items():
            if key != 'loss' and isinstance(value, torch.Tensor):
                metrics[f"train/{key}"] = value.detach().item()

        self.log(metrics)


# Image logging callback
class ImageLoggingCallback(TrainerCallback):
    def __init__(self, log_steps=1000, max_images=4):
        self.log_steps = log_steps
        self.max_images = max_images

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.log_steps == 0 and model is not None:
            # Create dummy batch for visualization
            batch_size = min(4, self.max_images)
            device = next(model.parameters()).device

            # Generate a dummy batch
            video = torch.rand(batch_size, 3, model.args.sequence_length,
                               model.args.resolution, model.args.resolution).to(device)

            # Generate reconstructions
            with torch.no_grad():
                frames, frames_rec, _, _ = model(video, log_image=True)

            # In a real implementation, you would log images using your preferred logging tool
            # For example with TensorBoard or Weights & Biases

            logging.info(f"Step {state.global_step}: Generated {batch_size} image reconstructions")

            # Print reconstructed image statistics for verification
            logging.info(
                f"Original frames shape: {frames.shape}, min: {frames.min().item():.4f}, max: {frames.max().item():.4f}")
            logging.info(
                f"Reconstructed frames shape: {frames_rec.shape}, min: {frames_rec.min().item():.4f}, max: {frames_rec.max().item():.4f}")


def main():
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Create default arguments for a quick test run
    output_dir = "./vqgan_output"
    os.makedirs(output_dir, exist_ok=True)

    # Default VQGANArguments - Even smaller for faster debugging
    vqgan_args = VQGANArguments(
        embedding_dim=32,  # Smaller to run faster
        n_codes=512,  # Smaller to run faster
        n_hiddens=64,  # Smaller to run faster
        resolution=32,  # Smaller to run faster
        sequence_length=4,  # Smaller to run faster
        use_dummy_data=True,
        dummy_data_samples=100,
        downsample=[2, 2, 2],  # Smaller downsampling for 32x32 resolution
    )

    # Default TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=3e-4,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        log_level='info',
        disable_tqdm=False,
        max_steps=20,  # Only run for 20 steps for this test
    )

    # Set random seed
    torch.manual_seed(training_args.seed)
    np.random.seed(training_args.seed)

    # Create datasets
    logging.info("Creating datasets...")
    train_dataset = DummyVideoDataset(
        sequence_length=vqgan_args.sequence_length,
        resolution=vqgan_args.resolution,
        num_samples=vqgan_args.dummy_data_samples
    )

    eval_dataset = DummyVideoDataset(
        sequence_length=vqgan_args.sequence_length,
        resolution=vqgan_args.resolution,
        num_samples=10  # Smaller eval dataset
    )

    # Initialize model
    logging.info("Initializing VQGAN model...")
    model = VQGAN(vqgan_args)

    # Print model size
    model_parameters = sum(p.numel() for p in model.parameters())
    logging.info(f"Model has {model_parameters:,} parameters")

    # Set up callbacks
    callbacks = [
        ImageLoggingCallback(log_steps=10, max_images=2),
    ]

    # Create trainer
    logging.info("Creating trainer...")
    trainer = VQGANTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        vqgan_args=vqgan_args
    )

    # Train model
    logging.info("Starting training...")
    try:
        trainer.train()
        logging.info("Training complete!")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

    # Save model
    logging.info("Saving model...")
    trainer.save_model(training_args.output_dir)

    # Run inference with a dummy input
    logging.info("Running inference with a dummy input...")
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        # Set up a smaller dummy input for inference testing
        dummy_input = torch.rand(1, 3, vqgan_args.sequence_length, vqgan_args.resolution, vqgan_args.resolution)
        dummy_input = dummy_input.to(device)

        # Run model
        try:
            frames, frames_rec, videos, videos_rec = model(dummy_input, log_image=True)

            logging.info(f"Input shape: {dummy_input.shape}")
            logging.info(f"Reconstructed video shape: {videos_rec.shape}")
            logging.info(f"Frames shape: {frames.shape}")
            logging.info(f"Reconstructed frames shape: {frames_rec.shape}")

            # Compute reconstruction error
            recon_error = F.mse_loss(videos, videos_rec).item()
            logging.info(f"Reconstruction error (MSE): {recon_error:.6f}")
        except Exception as e:
            logging.error(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()