import torch
import torch.nn as nn

from .networks import BaseNetwork, spectral_norm

class InpaintGenerator3D(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True, use_spectral_norm=True):
        super(InpaintGenerator3D, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad3d(4),
            spectral_norm(nn.Conv3d(
                in_channels=2, out_channels=64, kernel_size=(3, 9, 9), padding=1), use_spectral_norm),
            nn.InstanceNorm3d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv3d(
                in_channels=64, out_channels=128, kernel_size=(2, 4, 4), stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm3d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv3d(
                in_channels=128, out_channels=256, kernel_size=(2, 4, 4), stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm3d(256, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv3d(
                in_channels=256, out_channels=512, kernel_size=(2, 4, 4), stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm3d(512, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock3D(512, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose3d(
                in_channels=512, out_channels=256, kernel_size=(2, 4, 4), stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm3d(256, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose3d(
                in_channels=256, out_channels=128, kernel_size=(2, 4, 4), stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm3d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose3d(
                in_channels=128, out_channels=64, kernel_size=(2, 4, 4), stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm3d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad3d((4, 4, 4, 4, 0, 0)),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(3, 9, 9), padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)

        return x


class Discriminator3D(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator3D, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=64, kernel_size=(2, 4, 4), stride=2,
                                    padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(2, 4, 4), stride=2,
                                    padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(2, 4, 4), stride=2,
                                    padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(2, 4, 4), stride=1,
                                    padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=512, out_channels=1028, kernel_size=(2, 4, 4), stride=1,
                                    padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv6 = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=1028, out_channels=1, kernel_size=(2, 4, 4), stride=1,
                                    padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)

        if self.use_sigmoid:
            outputs = torch.sigmoid(conv6)
        else:
            outputs = conv6

        return outputs, [conv1, conv2, conv3, conv4, conv6]


class ResnetBlock3D(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock3D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad3d(dilation),
            spectral_norm(nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0,
                                    dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm3d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad3d(1),
            spectral_norm(nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0,
                                    dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm3d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out
