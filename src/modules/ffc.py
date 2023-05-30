import torch.nn.functional as F
from complexity import *


def actLayer(kind='relu'):
    if kind == 'tanh':
        return nn.Tanh()
    elif kind == 'sigmoid':
        return nn.Sigmoid()
    elif kind == 'relu':
        return nn.ReLU(inplace=True)
    elif kind == 'leaky':
        return nn.LeakyReLU(0.2, inplace=True)
    elif kind == 'elu':
        return nn.ELU(1.0, inplace=True)
    else:
        return nn.Identity()

def normLayer(channels, kind='bn', affine=True):
    if kind == 'bn':
        return nn.BatchNorm2d(channels, affine=affine)
    elif kind == 'in':
        return nn.InstanceNorm2d(channels, affine=affine)
    else:
        return nn.Identity(channels)

def upsample(scale_factor = 2, kind = 'interpolate'):
    if kind == 'interpolate':
        return nn.Upsample(scale_factor = scale_factor, mode="bilinear",align_corners=True)
    else:
        return nn.PixelShuffle(scale_factor)


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='kaiming', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                try:
                    nn.init.constant_(m.weight, 1)
                    nn.init.normal_(m.bias, 0.0001)
                except:
                    pass

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

    def print_networks(self, model_name):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()

        print('[Network %s] Total number of parameters : %.2f M' % (model_name, num_params / 1e6))
        print('-----------------------------------------------')

class SeperableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='reflect'):
        super(SeperableConv, self).__init__()
        self.depthConv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                   bias=bias, padding_mode=padding_mode)
        self.pointConv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=bias,
                                   padding_mode=padding_mode)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.depthConv(x)
        x = self.pointConv(x)

        return x

class MyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='reflect', kind='depthConv', norm_layer='',
                 activation=''):
        super(MyConv2d, self).__init__()
        if kind == 'depthConv':
            self.conv = SeperableConv(in_channels, out_channels, kernel_size, stride, padding,
                                      dilation, groups, bias, padding_mode=padding_mode)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                  dilation, groups, bias, padding_mode=padding_mode)

        self.norm = normLayer(kind=norm_layer, channels=out_channels)
        self.act = actLayer(kind=activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(self.norm(x))
        return x

class MyDeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='reflect', kind='depthConv', scale_mode='bilinear',
                 norm_layer='', activation=''):
        super(MyDeConv2d, self).__init__()
        if kind == 'depthConv':
            self.conv = SeperableConv(in_channels, out_channels, kernel_size, stride=1, padding=padding,
                                      dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

        self.norm = normLayer(kind=norm_layer, channels=out_channels)
        self.act = actLayer(kind=activation)

        self.scale_factor = stride
        self.scale_mode = scale_mode

    def forward(self, input):
        x = F.interpolate(input, scale_factor=self.scale_factor, mode=self.scale_mode)
        x = self.conv(x)
        x = self.act(self.norm(x))
        return x

# FU in paper
class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, use_spectral=False, norm_layer='bn', activation='relu'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        # kernel size was fixed to 1
        # because the global receptive field.
        if not use_spectral:
            self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                              kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        else:
            self.conv_layer = nn.utils.spectral_norm(
                torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False))
        self.norm = normLayer(kind=norm_layer, channels=out_channels * 2)
        self.act = actLayer(kind=activation)

        nn.init.kaiming_normal_(self.conv_layer.weight)

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        # The FFT of a real signal is Hermitian-symmetric, X[i_1, ..., i_n] = conj(X[-i_1, ..., -i_n])
        # so the full fftn() output contains redundant information.
        # rfftn() instead omits the negative frequencies in the last dimension.

        # (batch, c, h, w/2+1) complex number
        ffted = torch.fft.rfftn(x, s=(h, w), dim=(2, 3), norm='ortho')  # norm='ortho' making the real FFT orthonormal
        ffted = torch.cat([ffted.real, ffted.imag], dim=1)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.act(self.norm(ffted))

        ffted = torch.tensor_split(ffted, 2, dim=1)
        ffted = torch.complex(ffted[0], ffted[1])
        output = torch.fft.irfftn(ffted, s=(h, w), dim=(2, 3), norm='ortho')

        return output

class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, norm_layer='bn',
                 activation='relu'):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.enable_lfu = enable_lfu
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            normLayer(out_channels // 2, kind=norm_layer),
            actLayer(kind=activation)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, norm_layer=norm_layer, activation=activation)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups, norm_layer=norm_layer, activation=activation)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output

# light-weight version of original FFC
class NoFusionLFFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='reflect',
                 norm_layer='bn', activation='relu', enable_lfu=False, ratio_g_in=0.5, ratio_g_out=0.5, nc_reduce=2,
                 out_act=True):
        super(NoFusionLFFC, self).__init__()
        self.ratio_g_in = ratio_g_in
        self.ratio_g_out = ratio_g_out
        in_cg = int(in_channels * ratio_g_in)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_g_out)
        out_cl = out_channels - out_cg

        if in_cl > 0 and nc_reduce > 1:
            self.l_in_conv = nn.Sequential(
                nn.Conv2d(in_cl, in_cl // nc_reduce, kernel_size=1),
                normLayer(channels=in_cl // nc_reduce, kind=norm_layer),
                actLayer(kind=activation)
            )
        else:
            self.l_in_conv = nn.Identity()

        if out_cl > 0 and nc_reduce > 1:
            self.out_L_bn_act = nn.Sequential(
                nn.Conv2d(out_cl // nc_reduce, out_cl, kernel_size=1),
                normLayer(channels=out_cl, kind=norm_layer),
                actLayer(kind=activation if out_act else '')
            )
        elif out_cl > 0:
            self.out_L_bn_act = nn.Sequential(
                normLayer(channels=out_cl, kind=norm_layer),
                actLayer(kind=activation if out_act else '')
            )
        else:
            self.out_L_bn_act = nn.Identity()

        if in_cg > 0 and nc_reduce > 1:
            self.g_in_conv = self.g_in_conv = nn.Sequential(
                nn.Conv2d(in_cg, in_cg // nc_reduce, kernel_size=1),
                normLayer(channels=in_cg // nc_reduce, kind=norm_layer),
                actLayer(kind=activation)
            )
        else:
            self.g_in_conv = nn.Identity()

        if out_cg > 0 and nc_reduce > 1:
            self.out_G_bn_act = nn.Sequential(
                nn.Conv2d(out_cg // nc_reduce, out_cg, kernel_size=1),
                normLayer(channels=out_cg, kind=norm_layer),
                actLayer(kind=activation if out_act else '')
            )
        elif out_cg > 0:
            self.out_G_bn_act = nn.Sequential(
                normLayer(channels=out_cg, kind=norm_layer),
                actLayer(kind=activation if out_act else '')
            )
        else:
            self.out_G_bn_act = nn.Identity()

        module = nn.Identity if in_cl == 0 or out_cl == 0 else SeperableConv
        self.convl2l = module(in_cl // nc_reduce, out_cl // nc_reduce, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_mode)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else SeperableConv
        self.convl2g = module(in_cl // nc_reduce, out_cg // nc_reduce, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_mode)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else SeperableConv
        self.convg2l = module(in_cg // nc_reduce, out_cl // nc_reduce, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_mode)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform

        self.convg2g = module(in_cg // nc_reduce, out_cg // nc_reduce, stride=stride,
                              norm_layer=norm_layer, activation=activation, enable_lfu=enable_lfu)

        self.feats_dict = {}
        self.flops = 0

    def flops_count(self, module, input):
        if isinstance(module, nn.Module) and not isinstance(module, nn.Identity):
            if isinstance(input, torch.Tensor):
                # input_shape = input.shape[1:]
                flops = flop_counter(module, input)
                if flops != None:
                    self.flops += flops

    def get_flops(self):
        for m_name, input in self.feats_dict.items():
            module = getattr(self, m_name)
            self.flops_count(module, input)

        print(f'Total FLOPs : {self.flops:.5f} G')

        return self.flops

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        # self.feats_dict['l_in_conv'] = x_l
        # self.feats_dict['g_in_conv'] = x_g
        x_l, x_g = self.l_in_conv(x_l), self.g_in_conv(x_g)

        if self.ratio_g_out != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
            # self.feats_dict['convl2l'] = x_l
            # self.feats_dict['convg2l'] = x_g
            # self.feats_dict['out_L_bn_act'] = out_xl
            out_xl = self.out_L_bn_act(out_xl)
        if self.ratio_g_out != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)
            # self.feats_dict['convl2l'] = x_l
            # self.feats_dict['convg2l'] = x_g
            # self.feats_dict['out_G_bn_act'] = out_xg
            out_xg = self.out_G_bn_act(out_xg)

        return out_xl, out_xg

class FusedLFFCResNetBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, padding=1, norm_layer='bn', activation='relu', nc_reduce=2,
                 ratio_g_in=0.5, ratio_g_out=0.5):
        super(FusedLFFCResNetBlock, self).__init__()
        self.fusion = 0
        self.conv1 = NoFusionLFFC(in_channels=dim, out_channels=dim, kernel_size=kernel_size, padding=padding,
                                  norm_layer=norm_layer, activation=activation, out_act=True, nc_reduce=nc_reduce,
                                  ratio_g_in=0, ratio_g_out=ratio_g_out)

        # self.conv2 = NoFusionLFFC(in_channels=dim, out_channels=dim, kernel_size=kernel_size, padding=padding,
        #                           norm_layer=norm_layer, activation=activation, out_act=True, nc_reduce=nc_reduce,
        #                           ratio_g_in=ratio_g_in, ratio_g_out=ratio_g_out)
        #
        # self.conv3 = NoFusionLFFC(in_channels=dim, out_channels=dim, kernel_size=kernel_size, padding=padding,
        #                           norm_layer=norm_layer, activation=activation, out_act=False, nc_reduce=nc_reduce,
        #                          ratio_g_in=ratio_g_in, ratio_g_out=ratio_g_out)
                                 
        self.conv2 = TwoStreamLFFCResNetBlock(dim=dim,ratio_g_in=ratio_g_in,ratio_g_out=ratio_g_out,nc_reduce=4)
        self.conv3 = TwoStreamLFFCResNetBlock(dim=dim,ratio_g_in=ratio_g_in,ratio_g_out=ratio_g_out,nc_reduce=4)

    def get_flops(self):
        self.conv1.get_flops()
        self.conv2.get_flops()
        self.flops += self.conv1.flops + self.conv2.flops
        print(f'Total FLOPs : {self.flops:.5f} G')
        return self.flops

    def forward(self, x):
        res_x = x
        x = self.conv1(x)
        x = self.conv2(x)
        x_l, x_g = self.conv3(x)

        out_x = torch.cat([x_l,x_g], dim=1)

        out = out_x + res_x

        return out

class TwoStreamLFFCResNetBlock(nn.Module):
    def __init__(self,dim,kernel_size=3,padding=1,norm_layer='bn',activation='relu',nc_reduce=2,
                 ratio_g_in=0.5,ratio_g_out=0.5):
        super(TwoStreamLFFCResNetBlock, self).__init__()
        self.fusion = 0
        self.conv1 = NoFusionLFFC(in_channels=dim,out_channels=dim,kernel_size=kernel_size,padding=padding,
                                  norm_layer=norm_layer,activation=activation,out_act=True,nc_reduce=nc_reduce,
                                  ratio_g_in=ratio_g_in,ratio_g_out=ratio_g_out)
        self.conv2 = NoFusionLFFC(in_channels=dim, out_channels=dim, kernel_size=kernel_size,padding=padding,
                                  norm_layer=norm_layer, activation=activation, out_act=False, nc_reduce=nc_reduce,
                                  ratio_g_in=ratio_g_in,ratio_g_out=ratio_g_out)

    def get_flops(self):
        self.conv1.get_flops()
        self.conv2.get_flops()
        self.flops += self.conv1.flops + self.conv2.flops
        print(f'Total FLOPs : {self.flops:.5f} G')
        return self.flops


    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = self.conv1(x)
        x_l, x_g = self.conv2(x)

        out_x_l = x_l + id_l
        out_x_g = x_g + id_g
        return out_x_l, out_x_g



