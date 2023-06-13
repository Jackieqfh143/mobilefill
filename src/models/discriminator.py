import torch
import torch.nn as nn
import torch.nn.init as init
import random
import numpy as np
import functools
from complexity import *
from src.modules.attention import SEAttention
from src.modules.attention import MobileViTv2Attention

class SeperableConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1, groups=1,bias = True,padding_mode='reflect'):
        super(SeperableConv, self).__init__()
        self.depthConv = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups = in_channels,bias=bias, padding_mode=padding_mode)
        self.pointConv = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,groups=1,bias=bias, padding_mode=padding_mode)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self,x):
        x = self.depthConv(x)
        x = self.pointConv(x)

        return x

class MultidilatedConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size = 3, dilation_num=3, comb_mode='sum', equal_dim=True,
                 padding=1, min_dilation=1, use_depthwise=False, **kwargs):
        super().__init__()
        convs = []
        self.equal_dim = equal_dim
        assert comb_mode in ('cat_out', 'sum', 'cat_in', 'cat_both'), comb_mode
        if comb_mode in ('cat_out', 'cat_both'):
            self.cat_out = True
            if equal_dim:
                assert out_dim % dilation_num == 0
                out_dims = [out_dim // dilation_num] * dilation_num
                self.index = sum([[i + j * (out_dims[0]) for j in range(dilation_num)] for i in range(out_dims[0])], [])
            else:
                out_dims = [out_dim // 2 ** (i + 1) for i in range(dilation_num - 1)]
                out_dims.append(out_dim - sum(out_dims))
                index = []
                starts = [0] + out_dims[:-1]
                lengths = [out_dims[i] // out_dims[-1] for i in range(dilation_num)]
                for i in range(out_dims[-1]):
                    for j in range(dilation_num):
                        index += list(range(starts[j], starts[j] + lengths[j]))
                        starts[j] += lengths[j]
                self.index = index
                assert(len(index) == out_dim)
            self.out_dims = out_dims
        else:
            self.cat_out = False
            self.out_dims = [out_dim] * dilation_num

        if comb_mode in ('cat_in', 'cat_both'):
            if equal_dim:
                assert in_dim % dilation_num == 0
                in_dims = [in_dim // dilation_num] * dilation_num
            else:
                in_dims = [in_dim // 2 ** (i + 1) for i in range(dilation_num - 1)]
                in_dims.append(in_dim - sum(in_dims))
            self.in_dims = in_dims
            self.cat_in = True
        else:
            self.cat_in = False
            self.in_dims = [in_dim] * dilation_num

        conv_type = SeperableConv if use_depthwise else nn.Conv2d
        dilation = min_dilation
        for i in range(dilation_num):
            if isinstance(padding, int):
                cur_padding = padding * dilation
            else:
                cur_padding = padding[i]
            convs.append(conv_type(
                self.in_dims[i], self.out_dims[i], kernel_size, padding=cur_padding, dilation=dilation, **kwargs
            ))
            dilation *= 2
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        outs = []
        if self.cat_in:
            if self.equal_dim:
                x = x.chunk(len(self.convs), dim=1)
            else:
                new_x = []
                start = 0
                for dim in self.in_dims:
                    new_x.append(x[:, start:start+dim])
                    start += dim
                x = new_x
        for i, conv in enumerate(self.convs):
            if self.cat_in:
                input = x[i]
            else:
                input = x
            outs.append(conv(input))
        if self.cat_out:
            out = torch.cat(outs, dim=1)[:, self.index]
        else:
            out = sum(outs)
        return out

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,with_attn=False,nc_reduce=8):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//nc_reduce , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//nc_reduce , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,height,width  = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1) # B X (*W*H) X (C)
        proj_key =  self.key_conv(x).view(m_batchsize, -1, width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # B X (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,height,width)

        out = self.gamma*out + x

        if self.with_attn:
            return out,attention
        else:
            return out

class AttentionModule(nn.Module):
    def __init__(self,in_nc,use_spatial_att=True,reduction=8):
        super(AttentionModule, self).__init__()
        self.channel_attention = SEAttention(channel=in_nc,reduction=reduction)
        if use_spatial_att:
            self.spatial_attention = Self_Attn(in_dim=in_nc)
        else:
            self.spatial_attention = nn.Identity()

    def forward(self,x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x


class AttentionModule_V2(nn.Module):
    def __init__(self,in_nc,use_spatial_att=True):
        super(AttentionModule_V2, self).__init__()
        self.channel_attention = SEAttention(channel=in_nc)
        if use_spatial_att:
            self.spatial_attention = Self_Attn(in_dim=in_nc)
        else:
            self.spatial_attention = nn.Identity()

    def forward(self,input):
        x_ca = self.channel_attention(input)
        x_sa = self.spatial_attention(input)

        return input + x_ca + x_sa


class MobileAttentionModule(nn.Module):
    def __init__(self,in_nc,use_spatial_att=True):
        super(MobileAttentionModule, self).__init__()
        self.channel_attention = SEAttention(channel=in_nc)
        if use_spatial_att:
            self.spatial_attention = MobileViTv2Attention(d_model=in_nc)
        else:
            self.spatial_attention = nn.Identity()

    def forward(self,x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x

class LatentCodesDiscriminator(nn.Module):
    def __init__(self, style_dim, n_mlp):
        super().__init__()

        self.style_dim = style_dim

        layers = []
        for i in range(n_mlp-1):
            layers.append(
                nn.Linear(style_dim, style_dim)
            )
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(512, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, w):
        return self.mlp(w)



#augmented with self-attention layer
class MultidilatedNLayerDiscriminatorWithAtt(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)

            cur_model = []
            cur_model += [
                MultidilatedConv(nf_prev, nf, kernel_size=kw, stride=2, dilation_num=4,padding=[2, 3, 6, 12]),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]
            sequence.append(cur_model)

        nf_prev = nf

        # cur_model = []
        # cur_model += [
        #     AttentionModule(in_nc=nf_prev)
        # ]
        #
        # sequence.append(cur_model)

        nf = min(nf * 2, 512)

        cur_model = []
        cur_model += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]
        sequence.append(cur_model)

        cur_model = []
        cur_model += [
            AttentionModule(in_nc=nf)
        ]

        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 3):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        return act[-1], act[:-1]

class MultidilatedNLayerDiscriminatorWithAtt_v2(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=4, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)

            cur_model = []
            cur_model += [
                MultidilatedConv(nf_prev, nf, kernel_size=4, stride=2, dilation_num=4,padding=[2, 3, 6, 12]),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]
            sequence.append(cur_model)

        nf_prev = nf

        # cur_model = []
        # cur_model += [
        #     AttentionModule(in_nc=nf_prev)
        # ]
        #
        # sequence.append(cur_model)

        nf = min(nf * 2, 512)

        cur_model = []
        cur_model += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]
        sequence.append(cur_model)

        cur_model = []
        cur_model += [
            AttentionModule(in_nc=nf)
        ]

        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 3):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        return act[-1], act[:-1]

#augmented wit self-attention layer
class MultidilatedNLayerDiscriminatorWithAtt_UNet(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=4, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.n_layers = n_layers
        self.down_layer0 = nn.Sequential(nn.Conv2d(input_nc, ndf, kernel_size=5, stride=2, padding=2),
                     nn.LeakyReLU(0.2, True))

        nf = ndf
        for i in range(1,n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            model = nn.Sequential(
                MultidilatedConv(nf_prev, nf, kernel_size=3, stride=2, dilation_num=4, padding=[1, 2, 4, 8]),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            )
            setattr(self,f"down_layer{i}",model)

        nf = min(nf * 2, 512)
        self.att = AttentionModule(in_nc=nf)

        for i in range(n_layers):
            nf_out = nf // 2
            model = nn.Sequential(nn.ConvTranspose2d(in_channels=nf,out_channels=nf_out,kernel_size=3,stride=2,
                                                     padding=1,output_padding=1),
                                norm_layer(nf_out),
                                nn.LeakyReLU(0.2, True))
            setattr(self,f"up_layer{i}",model)
            nf = nf_out


        self.out = nn.Conv2d(nf, 1, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        for i in range(self.n_layers):
            x = getattr(self,f"down_layer{i}")(x)

        x = self.att(x)

        for i in range(self.n_layers):
            x = getattr(self,f"up_layer{i}")(x)

        x = self.out(x)

        return x, None

from math import floor, log2
from functools import partial
from linear_attention_transformer import ImageLinearAttention


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class Flatten(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
    def forward(self, x):
        return x.flatten(self.index)

class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return self.fn(x) * self.g


attn_and_ff = lambda chan: nn.Sequential(*[
    Residual(Rezero(ImageLinearAttention(chan, norm_queries = True))),
    Residual(Rezero(nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
])



def leaky_relu(p=0.2):
    return nn.LeakyReLU(p)

def double_conv(chan_in, chan_out):
    return nn.Sequential(
        nn.Conv2d(chan_in, chan_out, 3, padding=1),
        leaky_relu(),
        nn.Conv2d(chan_out, chan_out, 3, padding=1),
        leaky_relu()
    )

class DownBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = double_conv(input_channels, filters)
        self.down = nn.Conv2d(filters, filters, 3, padding = 1, stride = 2) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        unet_res = x

        if self.down is not None:
            x = self.down(x)

        x = x + res
        return x, unet_res

class UpBlock(nn.Module):
    def __init__(self, input_channels, filters):
        super().__init__()
        self.conv_res = nn.ConvTranspose2d(input_channels // 2, filters, 1, stride = 2)
        self.net = double_conv(input_channels, filters)
        self.up = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False)
        self.input_channels = input_channels
        self.filters = filters

    def forward(self, x, res):
        *_, h, w = x.shape
        conv_res = self.conv_res(x, output_size = (h * 2, w * 2))
        x = self.up(x)
        x = torch.cat((x, res), dim=1)
        x = self.net(x)
        x = x + conv_res
        return x

class UNetDiscriminator(nn.Module):
    def __init__(self, image_size, network_capacity = 16, transparent = False, fmap_max = 512):
        super().__init__()
        num_layers = int(log2(image_size) - 3)
        num_init_filters = 3 if not transparent else 4

        blocks = []
        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        filters[-1] = filters[-2]

        chan_in_out = list(zip(filters[:-1], filters[1:]))
        chan_in_out = list(map(list, chan_in_out))

        down_blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DownBlock(in_chan, out_chan, downsample = is_not_last)
            down_blocks.append(block)

            attn_fn = attn_and_ff(out_chan)
            attn_blocks.append(attn_fn)

        self.down_blocks = nn.ModuleList(down_blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)

        last_chan = filters[-1]

        self.to_logit = nn.Sequential(
            leaky_relu(),
            nn.AvgPool2d(image_size // (2 ** num_layers)),
            Flatten(1),
            nn.Linear(last_chan, 1)
        )

        self.conv = double_conv(last_chan, last_chan)

        dec_chan_in_out = chan_in_out[:-1][::-1]
        self.up_blocks = nn.ModuleList(list(map(lambda c: UpBlock(c[1] * 2, c[0]), dec_chan_in_out)))
        self.conv_out = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        b, *_ = x.shape

        residuals = []

        for (down_block, attn_block) in zip(self.down_blocks, self.attn_blocks):
            x, unet_res = down_block(x)
            residuals.append(unet_res)

            if attn_block is not None:
                x = attn_block(x)

        x = self.conv(x) + x
        enc_out = self.to_logit(x)

        for (up_block, res) in zip(self.up_blocks, residuals[:-1][::-1]):
            x = up_block(x, res)

        dec_out = self.conv_out(x)
        # return enc_out.squeeze(), dec_out

        return dec_out,None


from src.modules.legacy import *

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class StyleGAN_Discriminator(nn.Module):
    def __init__(self, size, channels_in=3, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], activate=False):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(channels_in, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )
        self.activate = activate

    def forward(self, x):
        out = self.convs(x)
        out = self.minibatch_discrimination(out, self.stddev_group, self.stddev_feat)
        out = self.final_conv(out)
        out = out.view(out.size(0), -1)
        out = self.final_linear(out)
        if self.activate:
            out = out.sigmoid()
        # return {"out": out}

        return out,None

    @staticmethod
    def minibatch_discrimination(x, stddev_group, stddev_feat):
        out = x
        batch, channel, height, width = out.shape
        group = min(batch, stddev_group)
        stddev = out.view(group, -1, stddev_feat, channel // stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        return out

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,):
        super().__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)

            cur_model = []
            cur_model += [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]
            sequence.append(cur_model)

        nf_prev = nf
        nf = min(nf * 2, 512)

        cur_model = []
        cur_model += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]
        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        return act[-1], act[:-1]

from src.modules.cnn_utils import *
import math
from src.modules.ffc import *
from src.modules.legacy import EqualLinear

class EESP(nn.Module):
    '''
    This class defines the EESP block, which is based on the following principle
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    '''

    def __init__(self, nIn, nOut, stride=1, k=4, r_lim=7, down_method='esp'): #down_method --> ['avg' or 'esp']
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
        expanded = self.conv_1x1_exp( # learn linear combinations using group point-wise convolutions
            self.br_after_cat( # apply batch normalization followed by activation function (PRelu in this case)
                torch.cat(output, 1) # concatenate the output of different branches
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
        # self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        # if reinf:
        #     self.inp_reinf = nn.Sequential(
        #         CBR(config_inp_reinf, config_inp_reinf, 3, 1),
        #         CB(config_inp_reinf, nout, 1, 1)
        #     )
        self.act = nn.PReLU(nout)

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

        return self.act(eesp_out)

class ToStyle(nn.Module):
    def __init__(self, in_channels, out_channels, activation='leaky'):
        super().__init__()
        self.conv = nn.Sequential(
            MyConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,stride=2,padding=1,activation=activation),
            MyConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,stride=2,padding=1,activation=activation),
            MyConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2,padding=1,activation=activation),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = EqualLinear(in_channels,out_channels,activation='fused_lrelu')

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x.flatten(start_dim=1))

        return x

class EESPDiscriminator(nn.Module):
    '''
    This class defines the ESPNetv3 architecture for encoder
    '''

    def __init__(self, input_nc = 3, output_nc = 1, latent_nc = 512, input_size = 256, s=1):
        '''
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()
        reps = [0, 3, 7, 3, 3]  # how many times EESP blocks should be repeated at each spatial level.
        channels = input_nc

        r_lim = [13, 11, 9, 7, 5, 3]  # receptive field at each spatial level
        K = [4]*len(r_lim) # No. of parallel branches at different levels

        base = 128 #base configuration
        config_len = int(math.log(input_size,2) - 3)
        config = [base] * config_len
        base_s = 0
        for i in range(config_len):
            if i== 0:
                base_s = int(base * s)
                base_s = math.ceil(base_s / K[0]) * K[0]
                config[i] = base if base_s > base else base_s
            else:
                config[i] = min(base_s * pow(2, i),latent_nc)

        global config_inp_reinf
        config_inp_reinf = input_nc
        self.input_reinforcement = True # True for the shortcut connection with input

        assert len(K) == len(r_lim), 'Length of branching factor array and receptive field array should be the same.'

        self.level1 = CBR(channels, config[0], 3, 2)  # 112 L1

        self.level2_0 = DownSampler(config[0], config[1], k=K[0], r_lim=r_lim[0], reinf=self.input_reinforcement)  # out = 56

        self.level3_0 = DownSampler(config[1], config[2], k=K[1], r_lim=r_lim[1], reinf=self.input_reinforcement) # out = 28

        self.level4_0 = DownSampler(config[2], config[3], k=K[2], r_lim=r_lim[2], reinf=self.input_reinforcement) #out = 14

        self.att = AttentionModule(in_nc=latent_nc)

        self.to_style = ToStyle(latent_nc, 1)

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
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''

        feats = {}
        out_l1 = self.level1(input)  # 112
        feats[str(out_l1.size(-1))] = out_l1
        if not self.input_reinforcement:
            del input
            input = None

        out_l2 = self.level2_0(out_l1, input)  # 56

        feats[str(out_l2.size(-1))] = out_l2
        out_l3_0 = self.level3_0(out_l2, input)  # down-sample
        feats[str(out_l3_0.size(-1))] = out_l3_0

        out_l4_0 = self.level4_0(out_l3_0, input)  # down-sample

        feats[str(out_l4_0.size(-1))] = out_l4_0

        out_x = self.att(out_l4_0)
        feats[str(out_l4_0.size(-1))] = out_x
        score = self.to_style(out_x)

        return score, feats

if __name__ == '__main__':
    from complexity import *
    from torchscan import summary
    from PIL import Image
    import glob
    import os
    from src.utils.util import cv2tensor
    x = torch.randn(1,3,256,256)
    model = MultidilatedNLayerDiscriminatorWithAtt(input_nc=3)
    print_network_params(model, "model")
    score = model(x)
    summary(model,[(3,256,256),])
    print()
    # model = MultidilatedNLayerDiscriminatorWithAtt_UNet(input_nc=3)
    # model = UNetDiscriminator(image_size=256)

    # src_dir = "/home/codeoops/CV/MobileFill/src/stylegan2/sample"
    # imgs_list = sorted(glob.glob(src_dir+"/*.png"))
    # model = StyleGAN_Discriminator(size=256)
    # model.load_state_dict(torch.load("/home/codeoops/CV/MobileFill/checkpoints/latest_dis.pth",map_location="cpu"))
    # state_dict = model.state_dict()
    # model.eval().requires_grad_(False)
    # print_network_params(model,"model")
    #
    # with torch.no_grad():
    #     for i in range(400):
    #         img = np.array(Image.open(imgs_list[i]))
    #         img_t = cv2tensor([img])
    #         score = torch.mean(model(img_t)[0])
    #         print("Img: ", imgs_list[i])
    #         print("score: ", score)

    # out_x,feats = model(x)
    # summary(model,(3,256,256))
    # flop_counter(model,x)


