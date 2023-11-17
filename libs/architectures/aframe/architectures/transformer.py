"""
In large part lifted from
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
but with 1d convolutions and arbitrary kernel sizes
"""

from typing import Callable, List, Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor


class ResNet(nn.Module):
    """
    transformer architecture
    """

    block = BasicBlock

    def __init__(
        self,
        num_ifos: int,
        layers: List[int],
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[List[str]] = None,
        norm_groups: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._norm_layer = get_norm_layer(norm_groups)

        self.inplanes = 64
        self.dilation = 1
        if stride_type is None:
            # each element in the tuple indicates if we should replace
            # the stride with a dilated convolution instead
            stride_type = ["stride"] * (len(layers) - 1)
        if len(stride_type) != (len(layers) - 1):
            raise ValueError(
                "'stride_type' should be None or a "
                "{}-element tuple, got {}".format(len(layers) - 1, stride_type)
            )

        self.groups = groups
        self.base_width = width_per_group

        # start with a basic conv-bn-relu-maxpool block
        # to reduce the dimensionality before the heavy
        # lifting starts
        self.conv1 = nn.Conv1d(
            num_ifos,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # now create layers of residual blocks where each
        # layer uses the same number of feature maps for
        # all its blocks (some power of 2 times 64).
        # Don't downsample along the time axis in the first
        # layer, but downsample in all the rest (either by
        # striding or dilating depending on the stride_type
        # argument)
        residual_layers = [self._make_layer(64, layers[0], kernel_size)]
        it = zip(layers[1:], stride_type)
        for i, (num_blocks, stride) in enumerate(it):
            block_size = 64 * 2 ** (i + 1)
            layer = self._make_layer(
                block_size,
                num_blocks,
                kernel_size,
                stride=2,
                stride_type=stride,
            )
            residual_layers.append(layer)
        self.residual_layers = nn.ModuleList(residual_layers)

        # Average pool over each feature map to create a
        # single value for each feature map that we'll use
        # in the fully connected head
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # use a fully connected layer to map from the
        # feature maps to the binary output that we need
        self.fc = nn.Linear(block_size * self.block.expansion, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        kernel_size: int = 3,
        stride: int = 1,
        stride_type: Literal["stride", "dilation"] = "stride",
    ) -> nn.Sequential:
        block = self.block
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if stride_type == "dilation":
            self.dilation *= stride
            stride = 1
        elif stride_type != "stride":
            raise ValueError("Unknown stride type {stride}")

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                kernel_size,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    kernel_size,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.residual_layers:
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)