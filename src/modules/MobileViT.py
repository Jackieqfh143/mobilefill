import torch
import torch.nn as nn

from einops import rearrange
from torch.nn import init
from src.modules.cnn_utils import *
from src.modules.attention import ECAAttention


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class EESPBlock(nn.Module):
    '''
    This class defines the EESP block, which is based on the following principle
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    '''

    def __init__(self, nIn, nOut, stride=1,d_rates=[1,2,4] , down_method='esp'): #down_method --> ['avg' or 'esp']
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
        # n = int(nOut / k)
        # n1 = nOut - (k - 1) * n
        # assert down_method in ['avg', 'esp'], 'One of these is suppported (avg or esp)'
        # assert n == n1, "n(={}) and n1(={}) should be equal for Depth-wise Convolution ".format(n, n1)
        self.proj_1x1 = CBR(nIn, 2 * nIn, 1, stride=1, groups=k)

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
        for i in range(k-1):
            d_rate = d_rates[i]
            self.spp_dw.append(CDilated_v2(2 * nIn, 2 * nIn, kSize=5, stride=stride, groups=nIn, d=d_rate))

        self.spp_dw.append(ca_attention)
        # self.se_unit = SEAttention(channel=nOut,reduction=16)

        # Performing a group convolution with K groups is the same as performing K point-wise convolutions
        self.conv_1x1_exp = CB(8 * nIn, nOut, 1, 1, groups=k)
        self.br_after_cat = BR(8 * nIn)
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
        # if self.stride == 2 and self.downAvg:
        #     return expanded
        #
        # # if dimensions of input and concatenated vector are the same, add them (RESIDUAL LINK)
        # if expanded.size() == input.size():
        expanded = expanded + input

        # Threshold the feature map using activation function (PReLU in this case)
        return self.module_act(expanded)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    def __init__(self, in_channels = 512 , image_size = (32,32), dims = [64, 80, 96], channels = 64, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        self.conv1 = conv_1x1_bn(in_channels, channels)
        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels, channels, 1, expansion))
        self.mv2.append(MV2Block(channels, channels, 1, expansion))
        self.mv2.append(MV2Block(channels, channels, 1, expansion))

        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels, kernel_size, patch_size, int(dims[0] * 2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels, kernel_size, patch_size, int(dims[1] * 4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels, kernel_size, patch_size, int(dims[2] * 4)))

        self.conv2 = conv_1x1_bn(channels, in_channels)
        #
        # self.pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(channels, latent_nc, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)
        x = self.mvit[0](x)

        x = self.mv2[1](x)
        x = self.mvit[1](x)

        x = self.mv2[2](x)
        x = self.mvit[2](x)
        x = self.conv2(x)
        #
        # x = self.pool(x).view(-1, x.shape[1])
        # x = self.fc(x)
        return x

class MobileViT_V2(nn.Module):
    def __init__(self, in_channels = 512 , image_size = (32,32), dims = [96, 120, 144], channels = 64, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        self.conv1 = conv_1x1_bn(in_channels, channels)
        self.mv2 = nn.ModuleList([])
        self.mv2.append(nn.Sequential(
            EESPBlock(channels, channels),
            EESPBlock(channels, channels),
            EESPBlock(channels, channels)))
        self.mv2.append(nn.Sequential(
            EESPBlock(channels,channels),
            EESPBlock(channels,channels),
            EESPBlock(channels, channels)))
        self.mv2.append(nn.Sequential(
            EESPBlock(channels, channels),
            EESPBlock(channels, channels),
            EESPBlock(channels, channels)))

        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels, kernel_size, patch_size, int(dims[0] * 2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels, kernel_size, patch_size, int(dims[1] * 4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels, kernel_size, patch_size, int(dims[2] * 4)))

        self.conv2 = conv_1x1_bn(channels, in_channels)
        #
        # self.pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(channels, latent_nc, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)
        x = self.mvit[0](x)

        x = self.mv2[1](x)
        x = self.mvit[1](x)

        x = self.mv2[2](x)
        x = self.mvit[2](x)
        x = self.conv2(x)
        #
        # x = self.pool(x).view(-1, x.shape[1])
        # x = self.fc(x)
        return x

class MobileViT_V3(nn.Module):
    def __init__(self, in_channels = 512 , image_size = (32,32), dims = [96, 120, 144], channels = 96, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        self.conv1 = conv_1x1_bn(in_channels, channels)
        self.mv2 = nn.ModuleList([])
        self.mv2.append(nn.Sequential(
            EESPBlock(channels, channels),
            EESPBlock(channels, channels),
            EESPBlock(channels, channels)))
        self.mv2.append(nn.Sequential(
            EESPBlock(channels,channels),
            EESPBlock(channels,channels),
            EESPBlock(channels, channels)))
        self.mv2.append(nn.Sequential(
            EESPBlock(channels, channels),
            EESPBlock(channels, channels),
            EESPBlock(channels, channels)))

        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels, kernel_size, patch_size, int(dims[0] * 2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels, kernel_size, patch_size, int(dims[1] * 4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels, kernel_size, patch_size, int(dims[2] * 4)))

        self.conv2 = conv_1x1_bn(channels, in_channels)
        #
        # self.pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(channels, latent_nc, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)
        x = self.mvit[0](x)

        x = self.mv2[1](x)
        x = self.mvit[1](x)

        x = self.mv2[2](x)
        x = self.mvit[2](x)
        x = self.conv2(x)
        #
        # x = self.pool(x).view(-1, x.shape[1])
        # x = self.fc(x)
        return x


if __name__ == '__main__':
    from complexity import *
    img = torch.randn(1, 512, 32, 32)
    vit = MobileViT()
    out = vit(img)
    print_network_params(vit, "mobile_vit")
    flop_counter(vit, img)
    # vit = mobilevit_xxs()
    # # out = vit(img)
    # print_network_params(vit, "vit_xxs")
    # # flop_counter(vit, img)
    # # print(out.shape)
    # # print(count_parameters(vit))
    #
    # vit = mobilevit_xs()
    # # out = vit(img)
    # print_network_params(vit, "vit_xs")
    # # flop_counter(vit, img)
    # #
    # vit = mobilevit_s()
    # # out = vit(img)
    # print_network_params(vit, "vit_s")
    # flop_counter(vit, img)