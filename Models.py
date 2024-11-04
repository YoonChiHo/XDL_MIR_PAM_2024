import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc

from torch.nn.parameter import Parameter
from torch.nn.init import constant_
from torch.nn.utils import spectral_norm
from torch.nn.modules.conv import ConvTranspose1d
from itertools import repeat
import numpy as np
from enum import Enum

class VisualizeSign(Enum):
    positive = 1
    absolute_value = 2
    negative = 3
    all = 4

def _generate_hold_kernel(in_channels, kernel_size, order):
    zoh_kernel_size = (kernel_size, kernel_size)

    # Zero-order hold kernel
    zoh_kernel = torch.Tensor(1, 1, *zoh_kernel_size)
    constant_(zoh_kernel, 1.0)
    tmp_kernel = zoh_kernel.clone()
    for i in range(order):
        tmp_kernel = F.conv2d(
            tmp_kernel, zoh_kernel, bias=None, stride=(1, 1),
            padding=((zoh_kernel_size[1]+1)//2,
                     (zoh_kernel_size[0]+1)//2),
            dilation=(1, 1), groups=1)
    return torch.repeat_interleave(tmp_kernel, in_channels, dim=0)

class Hold2d(nn.Module):
    def __init__(self, in_channels, zoh_kernel_size=2, order=0, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros'):
        super(Hold2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.zoh_kernel_size = zoh_kernel_size
        self.order = order
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = in_channels
        self.padding_mode = padding_mode

        kernel = _generate_hold_kernel(in_channels, self.zoh_kernel_size, self.order)
        self.kernel = Parameter(kernel, requires_grad=False)
        self.kernel_size = self.kernel.size()[2:]

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.kernel)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        expanded_padding = (self.kernel_size[1] // 2, (self.kernel_size[1]-1) // 2, self.kernel_size[0] // 2, (self.kernel_size[0]-1) // 2)
        input = F.pad(input, expanded_padding)

        return F.conv2d(input, self.kernel, self.bias, self.stride, self.padding, self.dilation, self.groups)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, label=0):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [  Hold2d(input_nc+label, zoh_kernel_size=2, order=0, stride=1, padding=0, bias=True, padding_mode='zeros'),
                   nn.ReflectionPad2d(3),
                   nn.Conv2d(input_nc+label, 64, 7, bias=False),
                   nn.InstanceNorm2d(64),
                   nn.ReLU(inplace=True)  ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  Hold2d(in_features, zoh_kernel_size=2, order=0, stride=1, padding=0, bias=True, padding_mode='zeros'),
                        nn.Conv2d(in_features, out_features, 3, stride=2, padding=1, bias=False),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True)  ]
            in_features = out_features     # 1: in = 128 / out = 256
            out_features = in_features*2   # 2: in = 256 / out = 512 
        
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [  ResidualBlock(in_features)  ]
        
        # Upsampling
        out_features = in_features//2 # in = 256 / out = 128
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        Hold2d(out_features, zoh_kernel_size=2, order=0, stride=1, padding=0, bias=False, padding_mode='zeros'),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True)  ]
            in_features = out_features      # 1: in = 128 / out = 64
            out_features = in_features//2   # 2: in = 64 / out = 32

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh()  ]
        
        self.model = nn.Sequential(*model)

    def forward(self, x, label_mask=[]):
        if len(label_mask[:]) > 0:
            out = torch.cat([x, label_mask], dim=1)
        else:
            out = x
        return self.model(out)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
         super(Discriminator, self).__init__()

         # A bunch of convolutions one after another
         model = [  nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)  ]  # [ 1 * 3 * 256 * 256 ] -> [ 1 * 64 * 128 * 128 ]
        
         model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                     nn.InstanceNorm2d(128),
                     nn.LeakyReLU(0.2, inplace=True)  ] # [ 1 * 64 * 128 * 128 ] -> [ 1 * 128 * 64 * 64 ]
        
         model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                     nn.InstanceNorm2d(256),
                     nn.LeakyReLU(0.2, inplace=True)  ] # [ 1 * 128 * 64 * 64 ] -> [ 1 * 256 * 32 * 32 ]

         model += [  nn.Conv2d(256, 512, 4, padding=1),
                     nn.InstanceNorm2d(512),
                     nn.LeakyReLU(0.2, inplace=True)  ] # [ 1 * 256 * 32 * 32 ] -> [ 1 * 512 * 31 * 31 ]
         
         # FCN classification layer
         model += [  nn.Conv2d(512, 1, 4, padding=1)  ] # [ 1 * 512 * 31 * 31 ] -> [ 1 * 1 * 30 * 30 ] (stride=1)

         self.model = nn.Sequential(*model)

    def forward(self, x):
         x = self.model(x)
         #print(x.shape)
         x = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
         # Average pooling and flattening
         return x


class gradcamModel(nn.Module):
    def __init__(self, base_model):
        super(gradcamModel, self).__init__()
        self.base_model = base_model  # 기존 모델
    def forward(self, x):
        x = self.base_model(x)  
        return torch.sum(x, dim=(2, 3)) #For gradcam
    
def normalize_attr(attr, sign, outlier_perc = 2, reduction_axis = 2):
    def _normalize_scale(attr, scale_factor):
        #if scale_factor == 0: scale_factor = 1e-5
        #assert scale_factor != 0, "Cannot normalize by scale factor = 0"
        if abs(scale_factor) < 1e-5:
            warnings.warn(
                "Attempting to normalize by value approximately 0, visualized results"
                "may be misleading. This likely means that attribution values are all"
                "close to 0."
            )
        attr_norm = attr / scale_factor
        return np.clip(attr_norm, -1, 1)

    def _cumulative_sum_threshold(values, percentile):
        # given values should be non-negative
        assert percentile >= 0 and percentile <= 100, (
            "Percentile for thresholding must be " "between 0 and 100 inclusive."
        )
        sorted_vals = np.sort(values.flatten())
        cum_sums = np.cumsum(sorted_vals)
        threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
        return sorted_vals[threshold_id]
    
    attr_combined = attr
    if reduction_axis is not None:
        attr_combined = np.sum(attr, axis=reduction_axis)

    # Choose appropriate signed values and rescale, removing given outlier percentage.
    if VisualizeSign[sign] == VisualizeSign.all:
        threshold = _cumulative_sum_threshold(np.abs(attr_combined), 100 - outlier_perc)
    elif VisualizeSign[sign] == VisualizeSign.positive:
        attr_combined = (attr_combined > 0) * attr_combined
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    elif VisualizeSign[sign] == VisualizeSign.negative:
        attr_combined = (attr_combined < 0) * attr_combined
        threshold = -1 * _cumulative_sum_threshold(
            np.abs(attr_combined), 100 - outlier_perc
        )
    elif VisualizeSign[sign] == VisualizeSign.absolute_value:
        attr_combined = np.abs(attr_combined)
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    else:
        raise AssertionError("Visualize Sign type is not valid.")
    return _normalize_scale(attr_combined, threshold)