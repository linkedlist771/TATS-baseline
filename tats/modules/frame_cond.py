# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .encoders import AbstractEncoder
import torchvision.models as models

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet34(nn.Module):
    def __init__(self, in_channels, output_shape, resnet_dim):
        super().__init__()
        self.in_channels = in_channels
        self.output_shape = output_shape
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, resnet_dim, 3, stride=2)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: [B, T, H, W, C]
        B, T, H, W, C = x.shape
        x = x.reshape(B*T, C, H, W)  # Reshape for 2D convolution
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Reshape back to include time dimension
        _, C, H, W = x.shape
        x = x.reshape(B, T, C, H, W)
        x = x.permute(0, 1, 3, 4, 2)  # [B, T, H, W, C]
        
        return x

class AddBroadcastPosEmbed(nn.Module):
    def __init__(self, shape, embd_dim):
        super().__init__()
        self.shape = shape
        self.n_dim = len(shape)
        
        for i, d in enumerate(shape):
            pos_embd = nn.Parameter(torch.zeros(1, *([1] * i), d, *([1] * (self.n_dim - i - 1)), embd_dim))
            nn.init.normal_(pos_embd, std=0.02)
            self.register_parameter(f'pos_embd_{i}', pos_embd)
    
    def forward(self, x):
        for i in range(self.n_dim):
            pos_embd = getattr(self, f'pos_embd_{i}')
            x = x + pos_embd
        return x

class FrameConditioner(AbstractEncoder):
    """Frame conditioning for video generation"""
    def __init__(self, resolution, n_cond_frames=1, embd_dim=240, quantize_interface=True):
        super().__init__()
        self.resolution = resolution
        self.n_cond_frames = n_cond_frames
        self.embd_dim = embd_dim
        self.quantize_interface = quantize_interface
        
        # Use pretrained ResNet34 from torchvision
        pretrained_resnet = models.resnet34(pretrained=True)
        
        # Keep most of the pretrained ResNet
        self.resnet_layers = nn.Sequential(
            pretrained_resnet.conv1,
            pretrained_resnet.bn1,
            pretrained_resnet.relu,
            pretrained_resnet.maxpool,
            pretrained_resnet.layer1,
            pretrained_resnet.layer2,
            pretrained_resnet.layer3,
            pretrained_resnet.layer4
        )
        
        # Add a projection layer to get the desired embedding dimension
        self.projection = nn.Conv2d(512, embd_dim, kernel_size=1)
        
        # Position embedding for frame features
        frame_cond_shape = (n_cond_frames, resolution // 32, resolution // 32)
        self.pos_embed = AddBroadcastPosEmbed(
            shape=frame_cond_shape,
            embd_dim=embd_dim
        )
        
        # Cache for frame features during inference
        self.frame_cond_cache = None
    
    def encode(self, x, include_embeddings=False, **kwargs):
        """
        Args:
            x: First frame(s) of the video [B, T, H, W, C] or [B, H, W, C]
        """
        if len(x.shape) == 4:  # [B, H, W, C]
            x = x.unsqueeze(1)  # Add time dimension [B, 1, H, W, C]
        
        # Take only the first n_cond_frames
        x = x[:, :self.n_cond_frames]
        
        # Extract features using pretrained ResNet
        B, T, H, W, C = x.shape
        x = x.reshape(B*T, C, H, W)  # [B*T, C, H, W]
        
        # ResNet expects channels first format and normalized inputs
        x = x.permute(0, 3, 1, 2)  # [B*T, C, H, W]
        
        # Normalize input for pretrained model
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        # Extract features
        features = self.resnet_layers(x)
        features = self.projection(features)  # Project to desired embedding dimension
        
        # Reshape back to include time dimension
        _, C, H, W = features.shape
        features = features.reshape(B, T, H, W, C)
        features = features.permute(0, 1, 2, 3, 4)  # [B, T, H, W, C]
        
        # Add positional embeddings
        features = self.pos_embed(features)
        
        # Flatten spatial and temporal dimensions for transformer input
        B, T, H, W, C = features.shape
        features_flat = features.reshape(B, T*H*W, C)
        
        if include_embeddings:
            # For the indices, we'll use a dummy tensor since we're not using discrete tokens
            # This is just to maintain compatibility with the VQGAN interface
            indices = torch.zeros(B, T*H*W, dtype=torch.long, device=features.device)
            return features_flat, indices
        
        return features_flat 