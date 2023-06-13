import random

import numpy as np
import torch
from torch.nn import init
from src.modules.cnn_utils import *
from src.modules.ffc import *
import math
from src.models.mapping_network import MappingNetwork
from src.modules.legacy import EqualLinear
from src.models.discriminator import Self_Attn, AttentionModule
from src.modules.attention import SEAttention, CrissCrossAttention, ECAAttention
from src.modules.MobileViT import MobileViT, MobileViT_V2, MobileViT_V3
from src.utils.diffaug import rand_cutout

# ============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"


# ============================================

class EESP(nn.Module):
    '''
    This class defines the EESP block, which is based on the following principle
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    '''

    def __init__(self, nIn, nOut, stride=1, k=4, r_lim=7, down_method='esp'):  # down_method --> ['avg' or 'esp']
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param stride: factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2
        :param k: # of parallel branches
        :param r_lim: A maximum value of receptive field allowed for EESP block
        :param down_method: Downsample or not (equivalent to say stride is 2 or not)
        '''
        super().__init__()
        self.stride = stride
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        assert down_method in ['avg', 'esp'], 'One of these is suppported (avg or esp)'
        assert n == n1, "n(={}) and n1(={}) should be equal for Depth-wise Convolution ".format(n, n1)
        self.proj_1x1 = CBR(nIn, n, 1, stride=1, groups=k)

        # (For convenience) Mapping between dilation rate and receptive field for a 3x3 kernel
        map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        self.k_sizes = list()
        for i in range(k):
            ksize = int(3 + 2 * i)
            # After reaching the receptive field limit, fall back to the base kernel size of 3 with a dilation rate of 1
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize)
        # sort (in ascending order) these kernel sizes based on their receptive field
        # This enables us to ignore the kernels (3x3 in our case) with the same effective receptive field in hierarchical
        # feature fusion because kernels with 3x3 receptive fields does not have gridding artifact.
        self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()
        for i in range(k):
            d_rate = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(CDilated(n, n, kSize=3, stride=stride, groups=n, d=d_rate))
        # Performing a group convolution with K groups is the same as performing K point-wise convolutions
        self.conv_1x1_exp = CB(nOut, nOut, 1, 1, groups=k)
        self.br_after_cat = BR(nOut)
        self.module_act = nn.PReLU(nOut)
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''

        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        # compute the output for each branch and hierarchically fuse them
        # i.e. Split --> Transform --> HFF
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            # HFF
            out_k = out_k + output[k - 1]
            output.append(out_k)
        # Merge
        expanded = self.conv_1x1_exp(  # learn linear combinations using group point-wise convolutions
            self.br_after_cat(  # apply batch normalization followed by activation function (PRelu in this case)
                torch.cat(output, 1)  # concatenate the output of different branches
            )
        )
        del output
        # if down-sampling, then return the concatenated vector
        # because Downsampling function will combine it with avg. pooled feature map and then threshold it
        if self.stride == 2 and self.downAvg:
            return expanded

        # if dimensions of input and concatenated vector are the same, add them (RESIDUAL LINK)
        if expanded.size() == input.size():
            expanded = expanded + input

        # Threshold the feature map using activation function (PReLU in this case)
        return self.module_act(expanded)


class EESP_V2(nn.Module):
    '''
    This class defines the EESP block, which is based on the following principle
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    '''

    def __init__(self, nIn, nOut, stride=1, d_rates=[1, 2, 4, 8],
                 down_method='esp'):  # down_method --> ['avg' or 'esp']
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param stride: factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2
        :param k: # of parallel branches
        :param r_lim: A maximum value of receptive field allowed for EESP block
        :param down_method: Downsample or not (equivalent to say stride is 2 or not)
        '''
        super().__init__()
        self.stride = stride
        k = len(d_rates)
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        assert down_method in ['avg', 'esp'], 'One of these is suppported (avg or esp)'
        assert n == n1, "n(={}) and n1(={}) should be equal for Depth-wise Convolution ".format(n, n1)
        self.proj_1x1 = CBR(nIn, n, 1, stride=1, groups=k)

        # (For convenience) Mapping between dilation rate and receptive field for a 3x3 kernel
        # map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        # self.k_sizes = list()
        # for i in range(k):
        #     ksize = int(3 + 2 * i)
        #     # After reaching the receptive field limit, fall back to the base kernel size of 3 with a dilation rate of 1
        #     ksize = ksize if ksize <= r_lim else 3
        #     self.k_sizes.append(ksize)
        # sort (in ascending order) these kernel sizes based on their receptive field
        # This enables us to ignore the kernels (3x3 in our case) with the same effective receptive field in hierarchical
        # feature fusion because kernels with 3x3 receptive fields does not have gridding artifact.
        # self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()
        for i in range(k):
            d_rate = d_rates[i]
            self.spp_dw.append(CDilated(n, n, kSize=3, stride=stride, groups=n, d=d_rate))
        # Performing a group convolution with K groups is the same as performing K point-wise convolutions
        self.conv_1x1_exp = CB(nOut, nOut, 1, 1, groups=k)
        self.br_after_cat = BR(nOut)
        self.module_act = nn.PReLU(nOut)
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''

        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        # compute the output for each branch and hierarchically fuse them
        # i.e. Split --> Transform --> HFF
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            # HFF
            out_k = out_k + output[k - 1]
            output.append(out_k)
        # Merge
        expanded = self.conv_1x1_exp(  # learn linear combinations using group point-wise convolutions
            self.br_after_cat(  # apply batch normalization followed by activation function (PRelu in this case)
                torch.cat(output, 1)  # concatenate the output of different branches
            )
        )
        del output
        # if down-sampling, then return the concatenated vector
        # because Downsampling function will combine it with avg. pooled feature map and then threshold it
        # if self.stride == 2 and self.downAvg:
        #     return expanded
        #
        # # if dimensions of input and concatenated vector are the same, add them (RESIDUAL LINK)
        # if expanded.size() == input.size():
        expanded = expanded + input

        # Threshold the feature map using activation function (PReLU in this case)
        return self.module_act(expanded)


# EESP with fourier unit
class EESP_V3(nn.Module):
    '''
    This class defines the EESP block, which is based on the following principle
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    '''

    def __init__(self, nIn, nOut, stride=1, d_rates=[1, 2, 8], down_method='esp'):  # down_method --> ['avg' or 'esp']
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param stride: factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2
        :param k: # of parallel branches
        :param r_lim: A maximum value of receptive field allowed for EESP block
        :param down_method: Downsample or not (equivalent to say stride is 2 or not)
        '''
        super().__init__()
        self.stride = stride
        k = len(d_rates) + 1
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        assert down_method in ['avg', 'esp'], 'One of these is suppported (avg or esp)'
        assert n == n1, "n(={}) and n1(={}) should be equal for Depth-wise Convolution ".format(n, n1)
        self.proj_1x1 = CBR(nIn, n, 1, stride=1, groups=k)

        # (For convenience) Mapping between dilation rate and receptive field for a 3x3 kernel
        # map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        # self.k_sizes = list()
        # for i in range(k):
        #     ksize = int(3 + 2 * i)
        #     # After reaching the receptive field limit, fall back to the base kernel size of 3 with a dilation rate of 1
        #     ksize = ksize if ksize <= r_lim else 3
        #     self.k_sizes.append(ksize)
        # sort (in ascending order) these kernel sizes based on their receptive field
        # This enables us to ignore the kernels (3x3 in our case) with the same effective receptive field in hierarchical
        # feature fusion because kernels with 3x3 receptive fields does not have gridding artifact.
        # self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()

        spectral_module = SpectralTransform(n, n, enable_lfu=False)
        for i in range(k - 1):
            d_rate = d_rates[i]
            self.spp_dw.append(CDilated(n, n, kSize=3, stride=stride, groups=n, d=d_rate))

        self.spp_dw.append(spectral_module)
        # Performing a group convolution with K groups is the same as performing K point-wise convolutions
        self.conv_1x1_exp = CB(nOut, nOut, 1, 1, groups=k)
        self.br_after_cat = BR(nOut)
        self.module_act = nn.PReLU(nOut)
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''

        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        # compute the output for each branch and hierarchically fuse them
        # i.e. Split --> Transform --> HFF
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            # HFF
            out_k = out_k + output[k - 1]
            output.append(out_k)
        # Merge
        expanded = self.conv_1x1_exp(  # learn linear combinations using group point-wise convolutions
            self.br_after_cat(  # apply batch normalization followed by activation function (PRelu in this case)
                torch.cat(output, 1)  # concatenate the output of different branches
            )
        )
        del output
        # if down-sampling, then return the concatenated vector
        # because Downsampling function will combine it with avg. pooled feature map and then threshold it
        # if self.stride == 2 and self.downAvg:
        #     return expanded
        #
        # # if dimensions of input and concatenated vector are the same, add them (RESIDUAL LINK)
        # if expanded.size() == input.size():
        expanded = expanded + input

        # Threshold the feature map using activation function (PReLU in this case)
        return self.module_act(expanded)


# EESP with SE unit
class EESP_V4(nn.Module):
    '''
    This class defines the EESP block, which is based on the following principle
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    '''

    def __init__(self, nIn, nOut, stride=1, d_rates=[1, 2, 4, 8],
                 down_method='esp'):  # down_method --> ['avg' or 'esp']
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param stride: factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2
        :param k: # of parallel branches
        :param r_lim: A maximum value of receptive field allowed for EESP block
        :param down_method: Downsample or not (equivalent to say stride is 2 or not)
        '''
        super().__init__()
        self.stride = stride
        k = len(d_rates) + 1
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        assert down_method in ['avg', 'esp'], 'One of these is suppported (avg or esp)'
        assert n == n1, "n(={}) and n1(={}) should be equal for Depth-wise Convolution ".format(n, n1)
        self.proj_1x1 = CBR(nIn, n, 1, stride=1, groups=k)

        # (For convenience) Mapping between dilation rate and receptive field for a 3x3 kernel
        # map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        # self.k_sizes = list()
        # for i in range(k):
        #     ksize = int(3 + 2 * i)
        #     # After reaching the receptive field limit, fall back to the base kernel size of 3 with a dilation rate of 1
        #     ksize = ksize if ksize <= r_lim else 3
        #     self.k_sizes.append(ksize)
        # sort (in ascending order) these kernel sizes based on their receptive field
        # This enables us to ignore the kernels (3x3 in our case) with the same effective receptive field in hierarchical
        # feature fusion because kernels with 3x3 receptive fields does not have gridding artifact.
        # self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()

        # att_module = CrissCrossAttention(in_dim = n)
        ca_attention = ECAAttention()
        for i in range(k - 1):
            d_rate = d_rates[i]
            self.spp_dw.append(CDilated_v2(n, n, kSize=5, stride=stride, groups=n, d=d_rate))

        self.spp_dw.append(ca_attention)
        # self.se_unit = SEAttention(channel=nOut,reduction=16)

        # Performing a group convolution with K groups is the same as performing K point-wise convolutions
        self.conv_1x1_exp = CB(nOut, nOut, 1, 1, groups=k)
        self.br_after_cat = BR(nOut)
        self.module_act = nn.PReLU(nOut)
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''

        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        # compute the output for each branch and hierarchically fuse them
        # i.e. Split --> Transform --> HFF
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            # HFF
            out_k = out_k + output[k - 1]
            output.append(out_k)

        # Merge
        expanded = self.conv_1x1_exp(  # learn linear combinations using group point-wise convolutions
            self.br_after_cat(  # apply batch normalization followed by activation function (PRelu in this case)
                torch.cat(output, 1)  # concatenate the output of different branches
            )
        )
        del output
        # if down-sampling, then return the concatenated vector
        # because Downsampling function will combine it with avg. pooled feature map and then threshold it
        # if self.stride == 2 and self.downAvg:
        #     return expanded
        #
        # # if dimensions of input and concatenated vector are the same, add them (RESIDUAL LINK)
        # if expanded.size() == input.size():
        expanded = expanded + input

        # Threshold the feature map using activation function (PReLU in this case)
        return self.module_act(expanded)


class DownSampler(nn.Module):
    '''
    Down-sampling fucntion that has three parallel branches: (1) avg pooling,
    (2) EESP block with stride of 2 and (3) efficient long-range connection with the input.
    The output feature maps of branches from (1) and (2) are concatenated and then additively fused with (3) to produce
    the final output.
    '''

    def __init__(self, nin, nout, k=4, r_lim=9, reinf=True):
        '''
            :param nin: number of input channels
            :param nout: number of output channels
            :param k: # of parallel branches
            :param r_lim: A maximum value of receptive field allowed for EESP block
            :param reinf: Use long range shortcut connection with the input or not.
        '''
        super().__init__()
        # nout_new = nout - nin
        self.eesp = EESP(nin, nout, stride=2, k=k, r_lim=r_lim, down_method='avg')
        self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        # if reinf:
        #     self.inp_reinf = nn.Sequential(
        #         CBR(config_inp_reinf, config_inp_reinf, 3, 1),
        #         CB(config_inp_reinf, nout, 1, 1)
        #     )
        # self.act = nn.PReLU(nout)

    def forward(self, input, input2=None):
        '''
        :param input: input feature map
        :return: feature map down-sampled by a factor of 2
        '''
        # avg_out = self.avg(input)
        eesp_out = self.eesp(input)
        # output = torch.cat([avg_out, eesp_out], 1)

        # if input2 is not None:
        #     #assuming the input is a square image
        #     # Shortcut connection with the input image
        #     w1 = avg_out.size(2)
        #     while True:
        #         input2 = F.avg_pool2d(input2, kernel_size=3, padding=1, stride=2)
        #         w2 = input2.size(2)
        #         if w2 == w1:
        #             break
        #     output = output + self.inp_reinf(input2)

        return eesp_out


class EESPNet(nn.Module):
    '''
    This class defines the ESPNetv2 architecture for the ImageNet classification
    '''

    def __init__(self, input_nc=4, output_nc=512, target_size=256, s=1):
        '''
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()
        reps = [0, 3, 7, 3, 3]  # how many times EESP blocks should be repeated at each spatial level.
        channels = input_nc

        r_lim = [13, 11, 9, 7, 5, 3]  # receptive field at each spatial level
        K = [4] * len(r_lim)  # No. of parallel branches at different levels

        base = 32  # base configuration
        config_len = int(math.log(target_size, 2) - 3)
        config = [base] * config_len
        base_s = 0
        for i in range(config_len):
            if i == 0:
                base_s = int(base * s)
                base_s = math.ceil(base_s / K[0]) * K[0]
                config[i] = base if base_s > base else base_s
            else:
                config[i] = base_s * pow(2, i)

        global config_inp_reinf
        config_inp_reinf = input_nc
        self.input_reinforcement = True  # True for the shortcut connection with input

        assert len(K) == len(r_lim), 'Length of branching factor array and receptive field array should be the same.'

        self.level1 = CBR(channels, config[0], 3, 2)  # 112 L1

        self.level2_0 = DownSampler(config[0], config[1], k=K[0], r_lim=r_lim[0],
                                    reinf=self.input_reinforcement)  # out = 56

        self.level3_0 = DownSampler(config[1], config[2], k=K[1], r_lim=r_lim[1],
                                    reinf=self.input_reinforcement)  # out = 28
        self.level3 = nn.ModuleList()
        for i in range(reps[1]):
            self.level3.append(EESP(config[2], config[2], stride=1, k=K[2], r_lim=r_lim[2]))

        self.level4_0 = DownSampler(config[2], config[3], k=K[2], r_lim=r_lim[2],
                                    reinf=self.input_reinforcement)  # out = 14
        self.level4 = nn.ModuleList()
        for i in range(reps[2]):
            self.level4.append(EESP(config[3], config[3], stride=1, k=K[3], r_lim=r_lim[3]))

        self.level5_0 = DownSampler(config[3], config[4], k=K[3], r_lim=r_lim[3])  # 7
        self.level5 = nn.ModuleList()
        for i in range(reps[3]):
            self.level5.append(EESP(config[4], config[4], stride=1, k=K[4], r_lim=r_lim[4]))

        # expand the feature maps using depth-wise convolution followed by group point-wise convolution
        # self.level5.append(CBR(config[4], config[4], 3, 1, groups=config[4]))
        # self.level5.append(CBR(config[4], config[5], 1, 1, groups=K[4]))

        self.out_affine = nn.Linear(config[4], output_nc)
        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input, p=0.2):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''

        feats = {}
        out_l1 = self.level1(input)  # 112
        feats[out_l1.size(-1)] = out_l1
        if not self.input_reinforcement:
            del input
            input = None

        out_l2 = self.level2_0(out_l1, input)  # 56

        feats[out_l2.size(-1)] = out_l2
        out_l3_0 = self.level3_0(out_l2, input)  # down-sample
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        feats[out_l3.size(-1)] = out_l3

        out_l4_0 = self.level4_0(out_l3, input)  # down-sample
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(out_l4_0)
            else:
                out_l4 = layer(out_l4)

        feats[out_l4.size(-1)] = out_l4

        out_l5_0 = self.level5_0(out_l4)  # down-sample
        for i, layer in enumerate(self.level5):
            if i == 0:
                out_l5 = layer(out_l5_0)
            else:
                out_l5 = layer(out_l5)

        feats[out_l5.size(-1)] = out_l5
        output_g = F.adaptive_avg_pool2d(out_l5, output_size=1)
        output_g = F.dropout(output_g, p=p, training=self.training)
        output_1x1 = output_g.view(output_g.size(0), -1)

        return self.out_affine(output_1x1), feats


class ToStyle(nn.Module):
    def __init__(self, in_channels, out_channels, activation='leaky'):
        super().__init__()
        self.conv = nn.Sequential(
            MyConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1,
                     activation=activation),
            MyConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1,
                     activation=activation),
            MyConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1,
                     activation=activation),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = EqualLinear(in_channels, out_channels, activation='fused_lrelu')

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x.flatten(start_dim=1))

        return x


class ToStyle_v2(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer='bn', activation='leaky'):
        super().__init__()
        self.conv = nn.Sequential(
            MyConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1,
                     norm_layer=norm_layer, activation=activation),
            MyConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1,
                     norm_layer=norm_layer, activation=activation),
            MyConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1,
                     norm_layer=norm_layer, activation=activation),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = EqualLinear(in_channels, out_channels, activation='fused_lrelu')

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x.flatten(start_dim=1))

        return x

class MiniToStyle(nn.Module):
    def __init__(self, in_channels, out_channels, activation='leaky'):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = in_channels // 4, kernel_size=1),
            MyConv2d(in_channels=in_channels // 4, out_channels=in_channels // 4, kernel_size=3, stride=2, padding=1,
                     activation=activation),
            MyConv2d(in_channels=in_channels // 4, out_channels=in_channels // 4, kernel_size=3, stride=2, padding=1,
                     activation=activation),
            MyConv2d(in_channels=in_channels // 4, out_channels=in_channels // 4, kernel_size=3, stride=2, padding=1,
                     activation=activation),
            nn.Conv2d(in_channels=in_channels // 4, out_channels=in_channels, kernel_size=1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = EqualLinear(in_channels, out_channels, activation='fused_lrelu')

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x.flatten(start_dim=1))
        return x


class LFFC_encoder(nn.Module):
    def __init__(self, input_nc, latent_nc=512, img_size=256, n_downsampling=4, n_blocks=4, ngf=32, norm_layer='bn',
                 activation='relu',
                 ratio_g_in=0.5, ratio_g_out=0.5, max_features=512, drop_out=0.5):
        super(LFFC_encoder, self).__init__()

        self.n_downsampling = n_downsampling
        self.n_blocks = n_blocks
        self.en_l0 = nn.Sequential(nn.ReflectionPad2d(2),
                                   NoFusionLFFC(in_channels=input_nc, out_channels=ngf, kernel_size=5, padding=0,
                                                ratio_g_in=0, ratio_g_out=0, nc_reduce=1, norm_layer=norm_layer,
                                                activation=activation))

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == 0:
                g_in = 0
            else:
                g_in = ratio_g_in
            model = NoFusionLFFC(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 ratio_g_in=g_in, ratio_g_out=ratio_g_out,
                                 nc_reduce=1, norm_layer=norm_layer, activation=activation)
            setattr(self, f"en_l{i + 1}", model)

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        # ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = TwoStreamLFFCResNetBlock(feats_num_bottleneck, ratio_g_in=ratio_g_in,
                                                    ratio_g_out=ratio_g_out, norm_layer=norm_layer,
                                                    activation=activation, nc_reduce=4)
            setattr(self, f"en_res_l{i}", cur_resblock)

        self.att = Self_Attn(in_dim=feats_num_bottleneck)

        # self.dropout = torch.nn.Dropout(p=drop_out)
        self.out_img_size = img_size // 2 ** n_downsampling
        self.to_style = ToStyle(latent_nc, latent_nc // 2)
        self.ws_style = EqualLinear(latent_nc, latent_nc // 2, activation="fused_lrelu")
        self.to_square = EqualLinear(latent_nc // 2, self.out_img_size * self.out_img_size, activation="fused_lrelu")

    def forward(self, x, ws):
        out_feats = {}
        for i in range(self.n_downsampling + 1):
            x = getattr(self, f"en_l{i}")(x)

            if i > 0 and i != self.n_downsampling:
                out_feats[x[0].size(-1)] = torch.cat(x, dim=1)

        for i in range(self.n_blocks):
            x = getattr(self, f"en_res_l{i}")(x)

        out_x = torch.cat(x, dim=1)

        out_x = self.att(out_x)

        mul_map = torch.ones_like(out_x) * 0.5
        mul_map = F.dropout(mul_map, training=True)
        ws = self.ws_style(ws)
        add_n = self.to_square(ws).view(-1, self.out_img_size, self.out_img_size).unsqueeze(1)
        add_n = F.interpolate(add_n, size=out_x.size()[-2:], mode='bilinear', align_corners=False)
        out_x = out_x * mul_map + add_n * (1 - mul_map)
        gs = self.to_style(out_x)

        return out_x, gs, ws, out_feats


class EESPNet_v10(nn.Module):
    '''
    This class defines the ESPNetv3 architecture for encoder
    '''

    def __init__(self, input_nc=4, output_nc=512, latent_nc=512, input_size=256, s=1):
        '''
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()
        base = 128  # base configuration
        config_len = int(math.log(input_size, 2) - 3)
        config = [base] * config_len
        for i in range(config_len):
            if i == 0:
                config[i] = base
            else:
                config[i] = min(base * pow(2, i), output_nc)

        self.level1_0 = CBR(input_nc, config[0], 3, 2)  # 112 L1

        self.level1 = nn.ModuleList()
        for i in range(3):
            self.level1.append(EESP_V3(config[0], config[0], stride=1, d_rates=[1, 2, 4]))

        self.level2_0 = DownSampler(config[0], config[1], k=4, r_lim=13)  # out = 56

        self.level2 = nn.ModuleList()
        for i in range(3):
            self.level2.append(EESP_V3(config[1], config[1], stride=1, d_rates=[1, 2, 4]))

        self.level3_0 = DownSampler(config[1], config[2], k=4, r_lim=13)  # out = 28
        self.level3 = nn.ModuleList()
        for i in range(3):
            self.level3.append(EESP_V3(config[2], config[2], stride=1, d_rates=[1, 2, 4]))

        self.level4_0 = DownSampler(config[2], config[3], k=4, r_lim=13)  # out = 14

        self.out_img_size = input_size // 2 ** 4
        self.to_style = ToStyle(latent_nc, latent_nc // 2)
        self.ws_style = EqualLinear(latent_nc, latent_nc // 2, activation="fused_lrelu")
        self.to_square = EqualLinear(latent_nc // 2, self.out_img_size * self.out_img_size, activation="fused_lrelu")

        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input, ws):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''

        feats = []
        out_l1_0 = self.level1_0(input)  # 112
        for i, layer in enumerate(self.level1):
            if i == 0:
                out_l1 = layer(out_l1_0)
            else:
                out_l1 = layer(out_l1)

        feats.append(out_l1)
        # feats[str(out_l1.size(-1))] = out_l1

        out_l2_0 = self.level2_0(out_l1)  # 56
        for i, layer in enumerate(self.level2):
            if i == 0:
                out_l2 = layer(out_l2_0)
            else:
                out_l2 = layer(out_l2)

        # feats[str(out_l2.size(-1))] = out_l2
        feats.append(out_l2)
        out_l3_0 = self.level3_0(out_l2)  # down-sample
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        feats.append(out_l3)
        # feats[str(out_l3.size(-1))] = out_l3

        out_l4_0 = self.level4_0(out_l3)  # down-sample

        # feats[str(out_l4_0.size(-1))] = out_l4_0

        out_x = out_l4_0
        mul_map = torch.ones_like(out_x) * 0.5
        mul_map = F.dropout(mul_map, training=True)
        ws = self.ws_style(ws)
        add_n = self.to_square(ws).view(-1, self.out_img_size, self.out_img_size).unsqueeze(1)
        add_n = F.interpolate(add_n, size=out_x.size()[-2:], mode='bilinear', align_corners=False)
        out_x = out_x * mul_map + add_n * (1 - mul_map)
        # feats[str(out_l4_0.size(-1))] = out_x
        feats.append(out_x)
        feats = feats[::-1]
        gs = self.to_style(out_x)

        ws = ws.unsqueeze(1).repeat(1, 11, 1)
        co_styles = []
        for i in range(11):
            co_styles.append(torch.cat((ws[:, i], gs), dim=-1))
        co_styles = torch.stack(co_styles, dim=1)

        return co_styles, feats


class EESPNet_v11(nn.Module):
    '''
    This class defines the ESPNetv4 architecture for encoder
    '''

    def __init__(self, input_nc=4, output_nc=512, latent_nc=512, input_size=256, s=1):
        '''
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()
        base = 128  # base configuration
        config_len = int(math.log(input_size, 2) - 3)
        config = [base] * config_len
        for i in range(config_len):
            if i == 0:
                config[i] = base
            else:
                config[i] = min(base * pow(2, i), output_nc)

        self.level1_0 = CBR(input_nc, config[0], 3, 2)  # 112 L1

        self.level1 = nn.ModuleList()
        for i in range(3):
            self.level1.append(EESP_V4(config[0], config[0], stride=1, d_rates=[1, 2, 4]))

        self.level2_0 = DownSampler(config[0], config[1], k=4, r_lim=13)  # out = 56

        self.level2 = nn.ModuleList()
        for i in range(3):
            self.level2.append(EESP_V4(config[1], config[1], stride=1, d_rates=[1, 2, 4]))

        self.level3_0 = DownSampler(config[1], config[2], k=4, r_lim=13)  # out = 28
        self.level3 = nn.ModuleList()
        for i in range(3):
            self.level3.append(EESP_V4(config[2], config[2], stride=1, d_rates=[1, 2, 4]))

        self.level4_0 = DownSampler(config[2], config[3], k=4, r_lim=13)  # out = 14

        self.out_img_size = input_size // 2 ** 4
        self.to_style = ToStyle(latent_nc, latent_nc // 2)
        self.ws_style = EqualLinear(latent_nc, latent_nc // 2, activation="fused_lrelu")
        self.to_square = EqualLinear(latent_nc // 2, self.out_img_size * self.out_img_size, activation="fused_lrelu")

        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input, ws):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''

        feats = []
        out_l1_0 = self.level1_0(input)  # 112
        for i, layer in enumerate(self.level1):
            if i == 0:
                out_l1 = layer(out_l1_0)
            else:
                out_l1 = layer(out_l1)

        feats.append(out_l1)
        # feats[str(out_l1.size(-1))] = out_l1

        out_l2_0 = self.level2_0(out_l1)  # 56
        for i, layer in enumerate(self.level2):
            if i == 0:
                out_l2 = layer(out_l2_0)
            else:
                out_l2 = layer(out_l2)

        # feats[str(out_l2.size(-1))] = out_l2
        feats.append(out_l2)
        out_l3_0 = self.level3_0(out_l2)  # down-sample
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        feats.append(out_l3)
        # feats[str(out_l3.size(-1))] = out_l3

        out_l4_0 = self.level4_0(out_l3)  # down-sample

        # feats[str(out_l4_0.size(-1))] = out_l4_0

        out_x = out_l4_0
        mul_map = torch.ones_like(out_x) * 0.5
        mul_map = F.dropout(mul_map, training=True)
        ws = self.ws_style(ws)
        add_n = self.to_square(ws).view(-1, self.out_img_size, self.out_img_size).unsqueeze(1)
        add_n = F.interpolate(add_n, size=out_x.size()[-2:], mode='bilinear', align_corners=False)
        out_x = out_x * mul_map + add_n * (1 - mul_map)
        # feats[str(out_l4_0.size(-1))] = out_x
        feats.append(out_x)
        feats = feats[::-1]
        gs = self.to_style(out_x)

        ws = ws.unsqueeze(1).repeat(1, 11, 1)
        co_styles = []
        for i in range(11):
            co_styles.append(torch.cat((ws[:, i], gs), dim=-1))
        co_styles = torch.stack(co_styles, dim=1)

        return co_styles, feats


class EESPNet_v12(nn.Module):
    '''
    This class defines the ESPNetv4 architecture for encoder
    '''

    def __init__(self, input_nc=4, output_nc=512, latent_nc=512, input_size=256, s=1):
        '''
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()
        base = 128  # base configuration
        config_len = int(math.log(input_size, 2) - 3)
        config = [base] * config_len
        for i in range(config_len):
            if i == 0:
                config[i] = base
            else:
                config[i] = min(base * pow(2, i), output_nc)

        self.level1_0 = CBR(input_nc, config[0], 3, 2)  # 112 L1

        self.level1 = nn.ModuleList()
        for i in range(3):
            self.level1.append(EESP_V4(config[0], config[0], stride=1, d_rates=[1, 2, 4]))

        self.level2_0 = DownSampler(config[0], config[1], k=4, r_lim=13)  # out = 56

        self.level2 = nn.ModuleList()
        for i in range(3):
            self.level2.append(EESP_V4(config[1], config[1], stride=1, d_rates=[1, 2, 4]))

        self.level3_0 = DownSampler(config[1], config[2], k=4, r_lim=13)  # out = 28
        self.level3 = nn.ModuleList()
        for i in range(3):
            self.level3.append(EESP_V4(config[2], config[2], stride=1, d_rates=[1, 2, 4]))

        self.level4_0 = DownSampler(config[2], config[3], k=4, r_lim=13)  # out = 14

        self.out_img_size = input_size // 2 ** 4
        self.to_style = ToStyle(latent_nc, latent_nc // 2)

        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input, ws, style_square):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''

        feats = []
        out_l1_0 = self.level1_0(input)  # 112
        for i, layer in enumerate(self.level1):
            if i == 0:
                out_l1 = layer(out_l1_0)
            else:
                out_l1 = layer(out_l1)

        feats.append(out_l1)
        # feats[str(out_l1.size(-1))] = out_l1

        out_l2_0 = self.level2_0(out_l1)  # 56
        for i, layer in enumerate(self.level2):
            if i == 0:
                out_l2 = layer(out_l2_0)
            else:
                out_l2 = layer(out_l2)

        # feats[str(out_l2.size(-1))] = out_l2
        feats.append(out_l2)
        out_l3_0 = self.level3_0(out_l2)  # down-sample
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        feats.append(out_l3)
        # feats[str(out_l3.size(-1))] = out_l3

        out_l4_0 = self.level4_0(out_l3)  # down-sample

        # feats[str(out_l4_0.size(-1))] = out_l4_0

        out_x = out_l4_0
        mul_map = torch.ones_like(out_x) * 0.5
        mul_map = F.dropout(mul_map, training=True)
        out_x = out_x * mul_map + style_square * (1 - mul_map)
        # feats[str(out_l4_0.size(-1))] = out_x
        feats.append(out_x)
        feats = feats[::-1]
        gs = self.to_style(out_x)

        ws = ws.unsqueeze(1).repeat(1, 11, 1)
        co_styles = []
        for i in range(11):
            co_styles.append(torch.cat((ws[:, i], gs), dim=-1))
        co_styles = torch.stack(co_styles, dim=1)

        return co_styles, feats


class EESPNet_v13(nn.Module):
    '''
    This class defines the ESPNetv4 architecture for encoder
    '''

    def __init__(self, input_nc=4, output_nc=512, latent_nc=512, input_size=256, s=1):
        '''
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()
        base = 128  # base configuration
        config_len = int(math.log(input_size, 2) - 3)
        config = [base] * config_len
        for i in range(config_len):
            if i == 0:
                config[i] = base
            else:
                config[i] = min(base * pow(2, i), output_nc)

        self.level1_0 = CBR(input_nc, config[0], 3, 2)  # 112 L1

        self.level1 = nn.ModuleList()
        for i in range(3):
            self.level1.append(EESP_V4(config[0], config[0], stride=1, d_rates=[1, 2, 4]))

        self.level2_0 = DownSampler(config[0], config[1], k=4, r_lim=13)  # out = 56

        self.level2 = nn.ModuleList()
        for i in range(3):
            self.level2.append(EESP_V4(config[1], config[1], stride=1, d_rates=[1, 2, 4]))

        self.level3_0 = DownSampler(config[1], config[2], k=4, r_lim=13)  # out = 28
        self.level3 = nn.ModuleList()
        for i in range(3):
            self.level3.append(EESP_V4(config[2], config[2], stride=1, d_rates=[1, 2, 4]))

        self.level4_0 = DownSampler(config[2], config[3], k=4, r_lim=13)  # out = 14

        self.out_img_size = input_size // 2 ** 4
        self.to_style = ToStyle_v2(latent_nc, latent_nc // 2)

        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input, ws, style_square):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''

        feats = []
        out_l1_0 = self.level1_0(input)  # 112
        for i, layer in enumerate(self.level1):
            if i == 0:
                out_l1 = layer(out_l1_0)
            else:
                out_l1 = layer(out_l1)

        feats.append(out_l1)
        # feats[str(out_l1.size(-1))] = out_l1

        out_l2_0 = self.level2_0(out_l1)  # 56
        for i, layer in enumerate(self.level2):
            if i == 0:
                out_l2 = layer(out_l2_0)
            else:
                out_l2 = layer(out_l2)

        # feats[str(out_l2.size(-1))] = out_l2
        feats.append(out_l2)
        out_l3_0 = self.level3_0(out_l2)  # down-sample
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        feats.append(out_l3)
        # feats[str(out_l3.size(-1))] = out_l3

        out_l4_0 = self.level4_0(out_l3)  # down-sample

        # feats[str(out_l4_0.size(-1))] = out_l4_0

        out_x = out_l4_0
        gs = self.to_style(out_x)
        # mul_map = torch.ones_like(out_x) * 0.5
        # mul_map = F.dropout(mul_map, training=True)
        # out_x = out_x * mul_map + style_square * (1 - mul_map)
        # feats[str(out_l4_0.size(-1))] = out_x
        if style_square.size() != out_x.size():
            style_square = F.interpolate(style_square, out_x.size()[-2:])
        feats.append(style_square)
        feats = feats[::-1]

        ws = ws.unsqueeze(1).repeat(1, 11, 1)
        co_styles = []
        for i in range(11):
            co_styles.append(torch.cat((ws[:, i], gs), dim=-1))
        co_styles = torch.stack(co_styles, dim=1)

        return co_styles, feats


class EESPNet_v14(nn.Module):
    '''
    This class defines the ESPNetv4 architecture for encoder
    '''

    def __init__(self, input_nc=4, output_nc=512, latent_nc=512, input_size=256, s=1):
        '''
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()
        base = 128  # base configuration
        config_len = int(math.log(input_size, 2) - 3)
        config = [base] * config_len
        for i in range(config_len):
            if i == 0:
                config[i] = base
            else:
                config[i] = min(base * pow(2, i), output_nc)

        self.level1_0 = CBR(input_nc, config[0], 3, 2)  # 112 L1

        self.level1 = nn.ModuleList()
        for i in range(3):
            self.level1.append(EESP_V4(config[0], config[0], stride=1, d_rates=[1, 2, 4]))

        self.level2_0 = DownSampler(config[0], config[1], k=4, r_lim=13)  # out = 56

        self.level2 = nn.ModuleList()
        for i in range(3):
            self.level2.append(EESP_V4(config[1], config[1], stride=1, d_rates=[1, 2, 4]))

        self.level3_0 = DownSampler(config[1], config[2], k=4, r_lim=13)  # out = 28
        self.level3 = nn.ModuleList()
        for i in range(3):
            self.level3.append(EESP_V4(config[2], config[2], stride=1, d_rates=[1, 2, 4]))

        self.level4_0 = DownSampler(config[2], config[3], k=4, r_lim=13)  # out = 14

        self.out_img_size = input_size // 2 ** 4
        self.to_style = ToStyle_v2(latent_nc, int(latent_nc * 0.25))

        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input, ws, style_square):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''

        feats = []
        out_l1_0 = self.level1_0(input)  # 112
        for i, layer in enumerate(self.level1):
            if i == 0:
                out_l1 = layer(out_l1_0)
            else:
                out_l1 = layer(out_l1)

        feats.append(out_l1)
        # feats[str(out_l1.size(-1))] = out_l1

        out_l2_0 = self.level2_0(out_l1)  # 56
        for i, layer in enumerate(self.level2):
            if i == 0:
                out_l2 = layer(out_l2_0)
            else:
                out_l2 = layer(out_l2)

        # feats[str(out_l2.size(-1))] = out_l2
        feats.append(out_l2)
        out_l3_0 = self.level3_0(out_l2)  # down-sample
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        feats.append(out_l3)
        # feats[str(out_l3.size(-1))] = out_l3

        out_l4_0 = self.level4_0(out_l3)  # down-sample

        # feats[str(out_l4_0.size(-1))] = out_l4_0

        out_x = out_l4_0
        gs = self.to_style(out_x)
        # mul_map = torch.ones_like(out_x) * 0.5
        # mul_map = F.dropout(mul_map, training=True)
        # out_x = out_x * mul_map + style_square * (1 - mul_map)
        # feats[str(out_l4_0.size(-1))] = out_x
        if style_square.size() != out_x.size():
            style_square = F.interpolate(style_square, out_x.size()[-2:])
        feats.append(style_square)
        feats = feats[::-1]

        ws = ws.unsqueeze(1).repeat(1, 11, 1)
        co_styles = []
        for i in range(11):
            co_styles.append(torch.cat((ws[:, i], gs), dim=-1))
        co_styles = torch.stack(co_styles, dim=1)

        return co_styles, feats


class EESPNet_v15(nn.Module):
    '''
    This class defines the ESPNetv4 architecture for encoder
    '''

    def __init__(self, input_nc=4, output_nc=512, latent_nc=512, input_size=256, s=1):
        '''
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()
        base = 128  # base configuration
        config_len = int(math.log(input_size, 2) - 3)
        config = [base] * config_len
        for i in range(config_len):
            if i == 0:
                config[i] = base
            else:
                config[i] = min(base * pow(2, i), output_nc)

        self.level1_0 = CBR(input_nc, config[0], 3, 2)  # 112 L1

        self.level1 = nn.ModuleList()
        for i in range(3):
            self.level1.append(EESP_V4(config[0], config[0], stride=1, d_rates=[1, 2, 4]))

        self.level2_0 = DownSampler(config[0], config[1], k=4, r_lim=13)  # out = 56

        self.level2 = nn.ModuleList()
        for i in range(3):
            self.level2.append(EESP_V4(config[1], config[1], stride=1, d_rates=[1, 2, 4]))

        self.level3_0 = DownSampler(config[1], config[2], k=4, r_lim=13)  # out = 28
        self.level3 = nn.ModuleList()
        for i in range(3):
            self.level3.append(EESP_V4(config[2], config[2], stride=1, d_rates=[1, 2, 4]))

        self.level4_0 = DownSampler(config[2], config[3], k=4, r_lim=13)  # out = 14

        self.out_img_size = input_size // 2 ** 4
        # self.to_style = ToStyle_v2(latent_nc, int(latent_nc * 0.25))

        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input, ws, style_square):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''

        feats = []
        out_l1_0 = self.level1_0(input)  # 112
        for i, layer in enumerate(self.level1):
            if i == 0:
                out_l1 = layer(out_l1_0)
            else:
                out_l1 = layer(out_l1)

        feats.append(out_l1)
        # feats[str(out_l1.size(-1))] = out_l1

        out_l2_0 = self.level2_0(out_l1)  # 56
        for i, layer in enumerate(self.level2):
            if i == 0:
                out_l2 = layer(out_l2_0)
            else:
                out_l2 = layer(out_l2)

        # feats[str(out_l2.size(-1))] = out_l2
        feats.append(out_l2)
        out_l3_0 = self.level3_0(out_l2)  # down-sample
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        feats.append(out_l3)
        # feats[str(out_l3.size(-1))] = out_l3

        out_l4_0 = self.level4_0(out_l3)  # down-sample

        # feats[str(out_l4_0.size(-1))] = out_l4_0

        out_x = out_l4_0
        # gs = self.to_style(out_x)
        # mul_map = torch.ones_like(out_x) * 0.5
        # mul_map = F.dropout(mul_map, training=True)
        # out_x = out_x * mul_map + style_square * (1 - mul_map)
        # feats[str(out_l4_0.size(-1))] = out_x
        # if style_square.size() != out_x.size():
        #     style_square = F.interpolate(style_square,out_x.size()[-2:])
        feats.append(out_x)
        feats = feats[::-1]

        ws = ws.unsqueeze(1).repeat(1, 11, 1)
        # co_styles = []
        # for i in range(11):
        #     co_styles.append(torch.cat((ws[:, i], gs), dim=-1))
        # co_styles = torch.stack(co_styles, dim=1)

        return ws, feats


class EESPNet_v16(nn.Module):
    '''
    This class defines the ESPNetv4 architecture for encoder
    '''

    def __init__(self, input_nc=4, output_nc=512, latent_nc=512, input_size=256, s=1):
        '''
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()
        base = 128  # base configuration
        config_len = int(math.log(input_size, 2) - 3)
        config = [base] * config_len
        for i in range(config_len):
            if i == 0:
                config[i] = base
            else:
                config[i] = min(base * pow(2, i), output_nc)

        self.level1_0 = CBR(input_nc, config[0], 3, 2)  # 112 L1

        self.level1 = nn.ModuleList()
        for i in range(3):
            self.level1.append(EESP_V4(config[0], config[0], stride=1, d_rates=[1, 2, 4]))

        self.level2_0 = DownSampler(config[0], config[1], k=4, r_lim=13)  # out = 56

        self.level2 = nn.ModuleList()
        for i in range(3):
            self.level2.append(EESP_V4(config[1], config[1], stride=1, d_rates=[1, 2, 4]))

        self.level3_0 = DownSampler(config[1], config[2], k=4, r_lim=13)  # out = 28
        self.level3 = nn.ModuleList()
        for i in range(3):
            self.level3.append(EESP_V4(config[2], config[2], stride=1, d_rates=[1, 2, 4]))

        # self.level4_0 = DownSampler(config[2], config[3], k=4, r_lim=13)  # out = 14

        self.out_img_size = input_size // 2 ** 4
        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input, ws, style_square):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''

        feats = []
        out_l1_0 = self.level1_0(input)  # 112
        for i, layer in enumerate(self.level1):
            if i == 0:
                out_l1 = layer(out_l1_0)
            else:
                out_l1 = layer(out_l1)

        feats.append(out_l1)
        # feats[str(out_l1.size(-1))] = out_l1

        out_l2_0 = self.level2_0(out_l1)  # 56
        for i, layer in enumerate(self.level2):
            if i == 0:
                out_l2 = layer(out_l2_0)
            else:
                out_l2 = layer(out_l2)

        # feats[str(out_l2.size(-1))] = out_l2
        feats.append(out_l2)
        out_l3_0 = self.level3_0(out_l2)  # down-sample
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        feats.append(out_l3)
        # feats[str(out_l3.size(-1))] = out_l3

        # out_l4_0 = self.level4_0(out_l3)  # down-sample
        #
        # feats[str(out_l4_0.size(-1))] = out_l4_0

        # out_x = out_l4_0
        if style_square.size(-1) != out_l3.size(-1) // 2:
            style_square = F.interpolate(style_square, (out_l3.size()[-2] // 2, out_l3.size()[-1] // 2))
        # mul_map = torch.ones_like(out_x) * 0.5
        # mul_map = F.dropout(mul_map, training=True)
        # out_x = out_x * mul_map + style_square * (1 - mul_map)
        #
        feats.append(style_square)
        feats = feats[::-1]

        ws = ws.unsqueeze(1).repeat(1, 11, 1)

        return ws, feats


class EESPNet_v17(nn.Module):
    '''
    This class defines the ESPNetv4 architecture for encoder
    '''

    def __init__(self, config, input_nc=4, output_nc=512, input_size=256, latent_nc=512, down_num=4):
        '''
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()
        # base = 128  # base configuration
        # config_len = int(math.log(input_size, 2) - 3)
        # config = [base] * config_len
        # for i in range(config_len):
        #     if i == 0:
        #         config[i] = base
        #     else:
        #         config[i] = min(base * pow(2, i), output_nc)

        self.level1_0 = CBR(input_nc, config[0], 3, 2)  # 112 L1

        self.level1 = nn.ModuleList()
        for i in range(3):
            self.level1.append(EESP_V4(config[0], config[0], stride=1, d_rates=[1, 2, 4]))

        self.level2_0 = DownSampler(config[0], config[1], k=4, r_lim=13)  # out = 56

        self.level2 = nn.ModuleList()
        for i in range(3):
            self.level2.append(EESP_V4(config[1], config[1], stride=1, d_rates=[1, 2, 4]))

        self.level3_0 = DownSampler(config[1], config[2], k=4, r_lim=13)  # out = 28
        self.level3 = nn.ModuleList()
        for i in range(3):
            self.level3.append(EESP_V4(config[2], config[2], stride=1, d_rates=[1, 2, 4]))

        self.level4_0 = DownSampler(config[2], config[3], k=4, r_lim=13)  # out = 14

        self.img_size = input_size // 2 ** down_num
        self.transformer = MobileViT(in_channels=config[3], image_size=(self.img_size, self.img_size))
        # self.to_style = ToStyle(config[3], latent_nc)
        self.to_square = EqualLinear(latent_nc, self.img_size * self.img_size, activation="fused_lrelu")
        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input, ws):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''

        feats = []
        out_l1_0 = self.level1_0(input)  # 112
        for i, layer in enumerate(self.level1):
            if i == 0:
                out_l1 = layer(out_l1_0)
            else:
                out_l1 = layer(out_l1)

        feats.append(out_l1)

        out_l2_0 = self.level2_0(out_l1)  # 56
        for i, layer in enumerate(self.level2):
            if i == 0:
                out_l2 = layer(out_l2_0)
            else:
                out_l2 = layer(out_l2)

        feats.append(out_l2)
        out_l3_0 = self.level3_0(out_l2)  # down-sample
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        feats.append(out_l3)

        out_l4_0 = self.level4_0(out_l3)  # down-sample

        mask = rand_cutout(out_l4_0, 0.5 * random.random())
        add_n = self.to_square(ws).view(-1, self.img_size, self.img_size).unsqueeze(1)
        add_n = F.interpolate(add_n, size=out_l4_0.size()[-2:], mode='bilinear', align_corners=False)
        out_x = out_l4_0 * mask + add_n * (1 - mask)

        out_x = self.transformer(out_x)
        feats.append(out_x)
        feats = feats[::-1]
        # gs = self.to_style(out_x)

        return ws, feats

    # def get_mask_ratio(self):
    #     if self.training:
    #         return 0.5 * random.random()
    #     else:
    #         return random.random()


class EESPNet_v18(nn.Module):
    '''
    This class defines the ESPNetv4 architecture for encoder
    '''

    def __init__(self, config, input_nc=4, output_nc=512, input_size=256, latent_nc=512, down_num=4):
        '''
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()
        # base = 128  # base configuration
        # config_len = int(math.log(input_size, 2) - 3)
        # config = [base] * config_len
        # for i in range(config_len):
        #     if i == 0:
        #         config[i] = base
        #     else:
        #         config[i] = min(base * pow(2, i), output_nc)

        self.level1_0 = CBR(input_nc, config[0], 3, 2)  # 112 L1

        self.level1 = nn.ModuleList()
        for i in range(3):
            self.level1.append(EESP_V4(config[0], config[0], stride=1, d_rates=[1, 2, 4]))

        self.level2_0 = DownSampler(config[0], config[1], k=4, r_lim=13)  # out = 56

        self.level2 = nn.ModuleList()
        for i in range(3):
            self.level2.append(EESP_V4(config[1], config[1], stride=1, d_rates=[1, 2, 4]))

        self.level3_0 = DownSampler(config[1], config[2], k=4, r_lim=13)  # out = 28
        self.level3 = nn.ModuleList()
        for i in range(3):
            self.level3.append(EESP_V4(config[2], config[2], stride=1, d_rates=[1, 2, 4]))

        self.level4_0 = DownSampler(config[2], config[3], k=4, r_lim=13)  # out = 14

        self.img_size = input_size // 2 ** down_num
        self.transformer = MobileViT_V2(in_channels=config[3], image_size=(self.img_size, self.img_size))
        # self.to_style = ToStyle(config[3], latent_nc)
        self.to_square = EqualLinear(latent_nc, self.img_size * self.img_size, activation="fused_lrelu")
        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input, ws):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''

        feats = []
        out_l1_0 = self.level1_0(input)  # 112
        for i, layer in enumerate(self.level1):
            if i == 0:
                out_l1 = layer(out_l1_0)
            else:
                out_l1 = layer(out_l1)

        feats.append(out_l1)

        out_l2_0 = self.level2_0(out_l1)  # 56
        for i, layer in enumerate(self.level2):
            if i == 0:
                out_l2 = layer(out_l2_0)
            else:
                out_l2 = layer(out_l2)

        feats.append(out_l2)
        out_l3_0 = self.level3_0(out_l2)  # down-sample
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        feats.append(out_l3)

        out_l4_0 = self.level4_0(out_l3)  # down-sample

        mask = rand_cutout(out_l4_0, random.random())
        add_n = self.to_square(ws).view(-1, self.img_size, self.img_size).unsqueeze(1)
        add_n = F.interpolate(add_n, size=out_l4_0.size()[-2:], mode='bilinear', align_corners=False)
        out_x = out_l4_0 * mask + add_n * (1 - mask)

        out_x = self.transformer(out_x)
        feats.append(out_x)
        feats = feats[::-1]
        # gs = self.to_style(out_x)

        return ws, feats

    # def get_mask_ratio(self):
    #     if self.training:
    #         return 0.5 * random.random()
    #     else:
    #         return random.random()

class EESPNet_v19(nn.Module):
    '''
    This class defines the ESPNetv4 architecture for encoder
    '''

    def __init__(self, config, input_nc=4, output_nc=512, input_size=256, latent_nc=512, down_num=4, mask_ratio=0.8):
        '''
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()
        # base = 128  # base configuration
        # config_len = int(math.log(input_size, 2) - 3)
        # config = [base] * config_len
        # for i in range(config_len):
        #     if i == 0:
        #         config[i] = base
        #     else:
        #         config[i] = min(base * pow(2, i), output_nc)

        self.mask_ratio = 0.8
        self.level1_0 = CBR(input_nc, config[0], 3, 2)  # 112 L1

        self.level1 = nn.ModuleList()
        for i in range(3):
            self.level1.append(EESP_V4(config[0], config[0], stride=1, d_rates=[1, 2, 4]))

        self.level2_0 = DownSampler(config[0], config[1], k=4, r_lim=13)  # out = 56

        self.level2 = nn.ModuleList()
        for i in range(3):
            self.level2.append(EESP_V4(config[1], config[1], stride=1, d_rates=[1, 2, 4]))

        self.level3_0 = DownSampler(config[1], config[2], k=4, r_lim=13)  # out = 28
        self.level3 = nn.ModuleList()
        for i in range(3):
            self.level3.append(EESP_V4(config[2], config[2], stride=1, d_rates=[1, 2, 4]))

        self.level4_0 = DownSampler(config[2], config[3], k=4, r_lim=13)  # out = 14

        self.img_size = input_size // 2 ** down_num
        self.transformer = MobileViT_V3(in_channels=config[3], image_size=(self.img_size, self.img_size))
        # self.to_style = ToStyle(config[3], latent_nc)
        self.to_square = EqualLinear(latent_nc, self.img_size * self.img_size, activation="fused_lrelu")
        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input, ws):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''

        feats = []
        out_l1_0 = self.level1_0(input)  # 112
        for i, layer in enumerate(self.level1):
            if i == 0:
                out_l1 = layer(out_l1_0)
            else:
                out_l1 = layer(out_l1)

        feats.append(out_l1)

        out_l2_0 = self.level2_0(out_l1)  # 56
        for i, layer in enumerate(self.level2):
            if i == 0:
                out_l2 = layer(out_l2_0)
            else:
                out_l2 = layer(out_l2)

        feats.append(out_l2)
        out_l3_0 = self.level3_0(out_l2)  # down-sample
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        feats.append(out_l3)

        out_l4_0 = self.level4_0(out_l3)  # down-sample

        # mask_ratio = random.random() if self.training else self.mask_ratio
        mask = rand_cutout(out_l4_0, random.random())
        add_n = self.to_square(ws).view(-1, self.img_size, self.img_size).unsqueeze(1)
        add_n = F.interpolate(add_n, size=out_l4_0.size()[-2:], mode='bilinear', align_corners=False)
        out_x = out_l4_0 * mask + add_n * (1 - mask)

        out_x = self.transformer(out_x)
        feats.append(out_x)
        feats = feats[::-1]
        # gs = self.to_style(out_x)

        return ws, feats

    # def get_mask_ratio(self):
    #     if self.training:
    #         return 0.5 * random.random()
    #     else:
    #         return random.random()

class EESPNet_v20(nn.Module):
    '''
    This class defines the ESPNetv4 architecture for encoder
    '''

    def __init__(self, config, input_nc=4, output_nc=512, input_size=256, latent_nc=512, down_num=4, mask_ratio=0.8):
        '''
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()
        self.mask_ratio = 0.8
        self.level1_0 = CBR(input_nc, config[0], 3, 2)  # 112 L1

        self.level1 = nn.ModuleList()
        for i in range(3):
            self.level1.append(EESP_V4(config[0], config[0], stride=1, d_rates=[1, 2, 4]))

        self.level2_0 = DownSampler(config[0], config[1], k=4, r_lim=13)  # out = 56

        self.level2 = nn.ModuleList()
        for i in range(3):
            self.level2.append(EESP_V4(config[1], config[1], stride=1, d_rates=[1, 2, 4]))

        self.level3_0 = DownSampler(config[1], config[2], k=4, r_lim=13)  # out = 28
        self.level3 = nn.ModuleList()
        for i in range(3):
            self.level3.append(EESP_V4(config[2], config[2], stride=1, d_rates=[1, 2, 4]))

        self.level4_0 = DownSampler(config[2], config[3], k=4, r_lim=13)  # out = 14

        self.img_size = input_size // 2 ** down_num
        self.transformer = MobileViT_V3(in_channels=config[3], channels=128, image_size=(self.img_size, self.img_size))
        self.to_style = ToStyle(config[3], latent_nc)
        self.to_square = EqualLinear(latent_nc, self.img_size * self.img_size, activation="fused_lrelu")
        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input,ws):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''

        feats = []
        out_l1_0 = self.level1_0(input)  # 112
        for i, layer in enumerate(self.level1):
            if i == 0:
                out_l1 = layer(out_l1_0)
            else:
                out_l1 = layer(out_l1)

        feats.append(out_l1)

        out_l2_0 = self.level2_0(out_l1)  # 56
        for i, layer in enumerate(self.level2):
            if i == 0:
                out_l2 = layer(out_l2_0)
            else:
                out_l2 = layer(out_l2)

        feats.append(out_l2)
        out_l3_0 = self.level3_0(out_l2)  # down-sample
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        feats.append(out_l3)

        out_l4_0 = self.level4_0(out_l3)  # down-sample

        mask = rand_cutout(out_l4_0, random.random())
        add_n = self.to_square(ws).view(-1, self.img_size, self.img_size).unsqueeze(1)
        add_n = F.interpolate(add_n, size=out_l4_0.size()[-2:], mode='bilinear', align_corners=False)
        out_x = out_l4_0 * mask + add_n * (1 - mask)

        out_x = self.transformer(out_x)
        feats.append(out_x)
        feats = feats[::-1]
        gs = self.to_style(out_x)

        return gs, feats

class EESPNet_v21(nn.Module):
    '''
    This class defines the ESPNetv4 architecture for encoder
    '''

    def __init__(self, config, input_nc=4, output_nc=512, input_size=256, latent_nc=512, down_num=5, mask_ratio=0.8):
        '''
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()
        self.mask_ratio = 0.8
        self.level1_0 = CBR(input_nc, config[0], 3, 2)  # 112 L1

        self.level1 = nn.ModuleList()
        for i in range(3):
            self.level1.append(EESP_V4(config[0], config[0], stride=1, d_rates=[1, 2, 4]))

        self.level2_0 = DownSampler(config[0], config[1], k=4, r_lim=13)  # out = 56

        self.level2 = nn.ModuleList()
        for i in range(3):
            self.level2.append(EESP_V4(config[1], config[1], stride=1, d_rates=[1, 2, 4]))

        self.level3_0 = DownSampler(config[1], config[2], k=4, r_lim=13)  # out = 28
        self.level3 = nn.ModuleList()
        for i in range(3):
            self.level3.append(EESP_V4(config[2], config[2], stride=1, d_rates=[1, 2, 4]))

        self.level4_0 = DownSampler(config[2], config[3], k=4, r_lim=13)  # out = 28
        self.level4 = nn.ModuleList()
        for i in range(3):
            self.level4.append(EESP_V4(config[3], config[3], stride=1, d_rates=[1, 2, 4]))

        self.level5_0 = DownSampler(config[3], config[4], k=4, r_lim=13)  # out = 14

        self.img_size = input_size // 2 ** down_num
        self.transformer = MobileViT_V3(in_channels=config[4], channels = 80,image_size=(self.img_size, self.img_size))
        self.to_style = ToStyle(config[4], latent_nc)
        self.to_square = EqualLinear(latent_nc, self.img_size * self.img_size, activation="fused_lrelu")
        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input,ws):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''

        feats = []
        out_l1_0 = self.level1_0(input)  # 112
        for i, layer in enumerate(self.level1):
            if i == 0:
                out_l1 = layer(out_l1_0)
            else:
                out_l1 = layer(out_l1)

        feats.append(out_l1)

        out_l2_0 = self.level2_0(out_l1)  # 56
        for i, layer in enumerate(self.level2):
            if i == 0:
                out_l2 = layer(out_l2_0)
            else:
                out_l2 = layer(out_l2)

        feats.append(out_l2)
        out_l3_0 = self.level3_0(out_l2)  # down-sample
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        feats.append(out_l3)

        out_l4_0 = self.level4_0(out_l3)  # down-sample
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(out_l4_0)
            else:
                out_l4 = layer(out_l4)

        feats.append(out_l4)

        out_l5_0 = self.level5_0(out_l4)  # down-sample

        mask = rand_cutout(out_l5_0, random.random())
        add_n = self.to_square(ws).view(-1, self.img_size, self.img_size).unsqueeze(1)
        add_n = F.interpolate(add_n, size=out_l5_0.size()[-2:], mode='bilinear', align_corners=False)
        out_x = out_l5_0 * mask + add_n * (1 - mask)

        out_x = self.transformer(out_x)
        feats.append(out_x)
        feats = feats[::-1]
        gs = self.to_style(out_x)

        return gs, feats

class EESPNet_v22(nn.Module):
    '''
    This class defines the ESPNetv4 architecture for encoder
    '''

    def __init__(self, config, input_nc=4, output_nc=512, input_size=256, latent_nc=512, down_num=4, mask_ratio=0.5):
        '''
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()
        self.mask_ratio = mask_ratio
        self.level1_0 = CBR(input_nc, config[0], 3, 2)  # 112 L1

        self.level1 = nn.ModuleList()
        for i in range(3):
            self.level1.append(EESP_V4(config[0], config[0], stride=1, d_rates=[1, 2, 4]))

        self.level2_0 = DownSampler(config[0], config[1], k=4, r_lim=13)  # out = 56

        self.level2 = nn.ModuleList()
        for i in range(3):
            self.level2.append(EESP_V4(config[1], config[1], stride=1, d_rates=[1, 2, 4]))

        self.level3_0 = DownSampler(config[1], config[2], k=4, r_lim=13)  # out = 28
        self.level3 = nn.ModuleList()
        for i in range(3):
            self.level3.append(EESP_V4(config[2], config[2], stride=1, d_rates=[1, 2, 4]))

        self.level4_0 = DownSampler(config[2], config[3], k=4, r_lim=13)  # out = 14

        self.img_size = input_size // 2 ** down_num
        self.transformer = MobileViT_V3(in_channels=config[3], image_size=(self.img_size, self.img_size))
        self.to_style = ToStyle(config[3], latent_nc)
        self.to_square = EqualLinear(latent_nc, self.img_size * self.img_size, activation="fused_lrelu")
        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input, ws):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''

        feats = []
        out_l1_0 = self.level1_0(input)  # 112
        for i, layer in enumerate(self.level1):
            if i == 0:
                out_l1 = layer(out_l1_0)
            else:
                out_l1 = layer(out_l1)

        feats.append(out_l1)

        out_l2_0 = self.level2_0(out_l1)  # 56
        for i, layer in enumerate(self.level2):
            if i == 0:
                out_l2 = layer(out_l2_0)
            else:
                out_l2 = layer(out_l2)

        feats.append(out_l2)
        out_l3_0 = self.level3_0(out_l2)  # down-sample
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        feats.append(out_l3)

        out_l4_0 = self.level4_0(out_l3)  # down-sample

        # mask_ratio = random.random() if self.training else self.mask_ratio
        mask = rand_cutout(out_l4_0, self.mask_ratio)
        add_n = self.to_square(ws).view(-1, self.img_size, self.img_size).unsqueeze(1)
        add_n = F.interpolate(add_n, size=out_l4_0.size()[-2:], mode='bilinear', align_corners=False)
        out_x = out_l4_0 * mask + add_n * (1 - mask)

        out_x = self.transformer(out_x)
        feats.append(out_x)
        feats = feats[::-1]
        gs = self.to_style(out_x)

        return gs, feats

    # def get_mask_ratio(self):
    #     if self.training:
    #         return 0.5 * random.random()
    #     else:
    #         return random.random()

class StyleEncoder(nn.Module):
    '''
    This class defines the ESPNetv4 architecture for encoder
    '''

    def __init__(self, config, input_nc=4,  input_size=256, latent_nc=512, down_num=4, mask_ratio=0.5):
        '''
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()
        self.mask_ratio = mask_ratio
        self.level1_0 = CBR(input_nc, config[0], 3, 2)  # 112 L1
        self.log_size = int(math.log(input_size, 2))
        self.n_latent = self.log_size * 2

        self.level1 = nn.ModuleList()
        for i in range(3):
            self.level1.append(EESP_V4(config[0], config[0], stride=1, d_rates=[1, 2, 4]))

        self.level2_0 = DownSampler(config[0], config[1], k=4, r_lim=13)  # out = 56

        self.level2 = nn.ModuleList()
        for i in range(3):
            self.level2.append(EESP_V4(config[1], config[1], stride=1, d_rates=[1, 2, 4]))

        self.level3_0 = DownSampler(config[1], config[2], k=4, r_lim=13)  # out = 28
        self.level3 = nn.ModuleList()
        for i in range(3):
            self.level3.append(EESP_V4(config[2], config[2], stride=1, d_rates=[1, 2, 4]))

        self.level4_0 = DownSampler(config[2], config[3], k=4, r_lim=13)  # out = 14

        self.img_size = input_size // 2 ** down_num
        self.latent_mlp = MappingNetwork(style_dim=latent_nc,n_layers=8)
        self.transformer = MobileViT_V3(in_channels=config[3], image_size=(self.img_size, self.img_size))
        self.style_module = nn.ModuleList()
        for i in range(down_num):
            self.style_module.append(MiniToStyle(config[i], latent_nc))
        self.to_square = EqualLinear(latent_nc, self.img_size * self.img_size, activation="fused_lrelu")
        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input, noise):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''

        feats = []
        deltas = []
        out_l1_0 = self.level1_0(input)  # 112
        for i, layer in enumerate(self.level1):
            if i == 0:
                out_l1 = layer(out_l1_0)
            else:
                out_l1 = layer(out_l1)

        feats.append(out_l1)
        delta1 = self.style_module[0](out_l1)
        delta1 = delta1.unsqueeze(1).repeat(1, 4, 1)
        deltas.append(delta1)

        out_l2_0 = self.level2_0(out_l1)  # 56
        for i, layer in enumerate(self.level2):
            if i == 0:
                out_l2 = layer(out_l2_0)
            else:
                out_l2 = layer(out_l2)

        feats.append(out_l2)
        delta2 = self.style_module[1](out_l2)
        delta2 = delta2.unsqueeze(1).repeat(1, 2, 1)
        deltas.append(delta2)

        out_l3_0 = self.level3_0(out_l2)  # down-sample
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        feats.append(out_l3)
        delta3 = self.style_module[2](out_l3)
        delta3 = delta3.unsqueeze(1).repeat(1, 2, 1)
        deltas.append(delta3)

        out_l4_0 = self.level4_0(out_l3)  # down-sample

        ws = self.latent_mlp(noise)
        mask = rand_cutout(out_l4_0, self.mask_ratio)
        add_n = self.to_square(ws).view(-1, self.img_size, self.img_size).unsqueeze(1)
        add_n = F.interpolate(add_n, size=out_l4_0.size()[-2:], mode='bilinear', align_corners=False)
        out_x = out_l4_0 * mask + add_n * (1 - mask)

        out_x = self.transformer(out_x)
        feats.append(out_x)
        feats = feats[::-1]
        gs = self.style_module[3](out_x)
        gs = gs.unsqueeze(1).repeat(1, self.n_latent, 1)
        init_idx = 2 * int(math.log(out_x.shape[-1],2))
        deltas = deltas[::-1]
        deltas = torch.cat(deltas,dim = 1)
        gs[:,init_idx:] += deltas
        return gs, deltas, feats

class RefineEncoder(nn.Module):
    def __init__(self, config = [64, 64, 128, 256, 512], input_nc=4, input_size=256, down_num=4):
        super().__init__()
        self.in_affine = CBR(input_nc, config[0], 3, 1)
        self.blocks = nn.ModuleList()
        for i in range(0, down_num):
            idx = min(len(config) - 1, i + 1)
            use_eesp = True if idx < len(config) - 1 else False
            self.blocks.append(EncoderBlock(in_dim = config[i], out_dim = config[idx], use_eesp = use_eesp))

        self.img_size = input_size // 2 ** down_num
        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        feats = []
        hidden = self.in_affine(input)
        for i, module in enumerate(self.blocks):
            hidden = module(hidden)
            feats.append(hidden)
        feats = feats[::-1]
        return feats[0], feats

class EncoderBlock(nn.Module):
    def __init__(self,in_dim, out_dim, use_eesp = True):
        super(EncoderBlock, self).__init__()
        self.down_block = DownSampler(in_dim, out_dim, k=4, r_lim=13)

        self.use_eesp = use_eesp

        self.layers = nn.ModuleList()
        for i in range(3):
            self.layers.append(EESP_V4(out_dim, out_dim, stride=1, d_rates=[1, 2, 4]))

    def forward(self, x):
        x = self.down_block(x)
        if not self.use_eesp:
            return x

        for i, module in enumerate(self.layers):
            x = module(x)
            return x


if __name__ == '__main__':
    from complexity import *

    channels = {
        4: 512,
        8: 512,
        16: 512,
        32: 256,
        64: 256,
        128: 128,
        256: 64,
        512: 64,
    }
    input = torch.Tensor(1, 4, 256, 256)
    ws = torch.randn(1, 512)
    style_square = torch.randn(1, 512, 16, 16)
    # en_channels = [v for k, v in channels.items() if k < input.size(-1)][::-1]

    en_channels = [128, 256, 256, 512]
    # model = EESPNet_v2(input_nc=4,output_nc=512,input_size=256)
    # model = EESPNet_v22(config=en_channels, input_nc=4, input_size=256)

    model = StyleEncoder(config = en_channels, input_nc=4, input_size=256)
    # model = LFFC_encoder(input_nc=4,latent_nc=512,n_downsampling=4,ngf=64)
    # summary(model,(4,256,256))
    print_network_params(model, "model")
    # print_network_params(model.out_affine,"out_affine")
    flop_counter(model,(input,ws))
    out = model(input, ws)

    en_channels = [64, 64, 128, 256, 512]
    model = RefineEncoder(config=en_channels, input_nc=4, input_size=256)
    print_network_params(model, "model")
    flop_counter(model, (input, ))
    refine_en_out = model(input)

    # print('Output size')
    # print(out.size())
