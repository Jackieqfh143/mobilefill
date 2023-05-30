import torch
from src.utils.noise_manager import NoiseManager
from src.modules.styled_conv2d import StyledConv2d
from src.modules.constant_input import ConstantInput
from src.modules.multichannel_image import MultichannelIamge
from src.modules.modulated_conv2d import ModulatedDWConv2d,ModulatedConv2d
from src.modules.mobile_synthesis_block import MobileSynthesisBlock,MobileSynthesisBlock_withFFC,\
    MobileSynthesisBlock_v2,MobileSynthesisBlock_v3,MobileSynthesisBlock_v4,MobileSynthesisBlock_v5
from src.modules.idwt_upsample import DWTInverse
from src.modules.ffc import actLayer
from src.modules.attention import ParallelContextualAttention,CAttention,ContextualAttentionModule
from src.models.discriminator import Self_Attn,AttentionModule
import torch.nn.functional as F
from complexity import *
from torchscan import summary
from src.modules.legacy import *

class StyleEncoder(nn.Module):
    def __init__(self,
                 style_dim = 256,
                 channels=[128, 256, 512],
                 blur_kernel=[1, 3, 3, 1],
                 device='cuda',
                 ):
        super(StyleEncoder, self).__init__()
        self.device = device
        self.input = ConstantInput(channels[0])
        self.conv1 = StyledConv(
            channels[0], channels[0], 3, style_dim, blur_kernel=blur_kernel
        )
        self.layers = nn.ModuleList()
        channels_in = channels[0]
        for i, channels_out in enumerate(channels[1:]):
            self.layers.append(
                MobileSynthesisBlock(
                    channels_in,
                    channels_out,
                    style_dim,
                    3,
                    conv_module=ModulatedDWConv2d,
                )
            )
            channels_in = channels_out


    def forward(self, style):
        hidden = self.input(style)
        _noise = torch.randn(1, 1, hidden.size(-2), hidden.size(-1)).to(style.device)
        hidden = self.conv1(hidden, style, noise=_noise)

        for i, m in enumerate(self.layers):
            start_style_idx = m.wsize() * i + 1
            end_style_idx = m.wsize() * i + m.wsize() + 1
            shape = [2, 1, 1, 2 ** (i + 3), 2 ** (i + 3)]
            _noise = torch.randn(*shape).to(style.device)
            hidden, freq = m(hidden, style if len(style.shape) == 2 else style[:, start_style_idx:end_style_idx, :],
                             noise=_noise)

        return hidden

class MobileSynthesisNetwork(nn.Module):
    def __init__(
            self,
            style_dim,
            channels = [512, 512, 512, 512, 256, 128, 64],
            device = 'cuda',
            out_act = 'tanh',
            trace_model = False,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.device = device

        self.layers = nn.ModuleList()
        channels_in = channels[0]
        for i, channels_out in enumerate(channels[1:]):
            self.layers.append(
                MobileSynthesisBlock_withFFC(
                    channels_in,
                    channels_out,
                    style_dim,
                    3,
                    conv_module = ModulatedDWConv2d,
                    # use_spatial_att = False if i >=2 else True,
                    use_spatial_att = False
                )
            )
            channels_in = channels_out

        self.trace_model = trace_model
        self.idwt = DWTInverse(mode="zero", wave="db1",trace_model=self.trace_model)
        self.out_act = actLayer(kind=out_act)

    def forward(self,style, en_feats,noise=None):
        out = {"noise": [], "freq": [], "img": None,
               "en_feats": en_feats}
        noise = NoiseManager(noise, self.device, self.trace_model)

        hidden = en_feats[0]

        for i, m in enumerate(self.layers):
            start_style_idx = m.wsize()*i + 1
            end_style_idx = m.wsize()*i + m.wsize() + 1
            out["noise"].append(noise(2 * hidden.size(-1), 2))
            hidden, freq = m(hidden,style if len(style.shape)==2 else style[:, start_style_idx:end_style_idx, :],
                             en_feats[i+1],
                             noise=out["noise"][-1])

            out["freq"].append(freq)

        out["img"] = self.dwt_to_img(out["freq"][-1])

        out["img"] = self.out_act(out["img"])

        return out

    def dwt_to_img(self, img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return self.idwt((low, [high]))

    def wsize(self):
        return len(self.layers) * self.layers[0].wsize() + 2

class MobileSynthesisNetwork_v2(nn.Module):
    def __init__(
            self,
            style_dim,
            channels = [512, 512, 512, 512, 256, 128, 64],
            device = 'cuda',
            out_act = 'tanh',
            trace_model = False,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.device = device

        self.layers = nn.ModuleList()
        channels_in = channels[0]
        for i, channels_out in enumerate(channels[1:]):
            self.layers.append(
                MobileSynthesisBlock_v2(
                    channels_in,
                    channels_out,
                    style_dim,
                    3,
                    conv_module = ModulatedDWConv2d,
                    # use_spatial_att = False if i >=2 else True,
                    use_spatial_att = False
                )
            )
            channels_in = channels_out

        self.trace_model = trace_model
        self.idwt = DWTInverse(mode="zero", wave="db1",trace_model=self.trace_model)
        self.out_act = actLayer(kind=out_act)

    def forward(self,style, en_feats,noise=None):
        out = {"noise": [], "freq": [], "img": None,
               "en_feats": en_feats}
        noise = NoiseManager(noise, self.device, self.trace_model)

        hidden = en_feats[0]

        for i, m in enumerate(self.layers):
            start_style_idx = m.wsize()*i + 1
            end_style_idx = m.wsize()*i + m.wsize() + 1
            out["noise"].append(noise(2 * hidden.size(-1), 2))
            hidden, freq = m(hidden,style if len(style.shape)==2 else style[:, start_style_idx:end_style_idx, :],
                             en_feats[i+1],
                             noise=out["noise"][-1])

            out["freq"].append(freq)

        out["img"] = self.dwt_to_img(out["freq"][-1])

        out["img"] = self.out_act(out["img"])

        return out

    def dwt_to_img(self, img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return self.idwt((low, [high]))

    def wsize(self):
        return len(self.layers) * self.layers[0].wsize() + 2

class MobileSynthesisNetwork_v3(nn.Module):
    def __init__(
            self,
            style_dim,
            channels = [512, 512, 512, 512, 256, 128, 64],
            device = 'cuda',
            out_act = 'tanh',
            trace_model = False,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.device = device

        self.layers = nn.ModuleList()
        channels_in = channels[0]
        for i, channels_out in enumerate(channels[1:]):
            self.layers.append(
                MobileSynthesisBlock_v3(
                    channels_in,
                    channels_out,
                    style_dim,
                    3,
                    conv_module = ModulatedDWConv2d,
                    # use_spatial_att = False if i >=2 else True,
                    use_spatial_att = False
                )
            )
            channels_in = channels_out

        self.trace_model = trace_model
        self.idwt = DWTInverse(mode="zero", wave="db1",trace_model=self.trace_model)
        self.out_act = actLayer(kind=out_act)

    def forward(self,style, en_feats,noise=None):
        out = {"noise": [], "freq": [], "img": None,
               "en_feats": en_feats}
        noise = NoiseManager(noise, self.device, self.trace_model)

        hidden = en_feats[0]

        for i, m in enumerate(self.layers):
            start_style_idx = m.wsize()*i + 1
            end_style_idx = m.wsize()*i + m.wsize() + 1
            out["noise"].append(noise(2 * hidden.size(-1), 2))
            hidden, freq = m(hidden,style if len(style.shape)==2 else style[:, start_style_idx:end_style_idx, :],
                             en_feats[i+1],
                             noise=out["noise"][-1])

            out["freq"].append(freq)

        out["img"] = self.dwt_to_img(out["freq"][-1])

        out["img"] = self.out_act(out["img"])

        return out

    def dwt_to_img(self, img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return self.idwt((low, [high]))

    def wsize(self):
        return len(self.layers) * self.layers[0].wsize() + 2

class MobileSynthesisNetwork_v4(nn.Module):
    def __init__(
            self,
            style_dim,
            channels = [512, 512, 512, 512, 256, 128, 64],
            device = 'cuda',
            out_act = 'tanh',
            trace_model = False,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.device = device

        self.layers = nn.ModuleList()
        channels_in = channels[0]
        for i, channels_out in enumerate(channels[1:]):
            self.layers.append(
                MobileSynthesisBlock_v4(
                    channels_in,
                    channels_out,
                    style_dim,
                    3,
                    conv_module = ModulatedDWConv2d,
                    # use_spatial_att = False if i >=2 else True,
                    use_spatial_att = False
                )
            )
            channels_in = channels_out

        self.trace_model = trace_model
        self.idwt = DWTInverse(mode="zero", wave="db1",trace_model=self.trace_model)
        self.out_act = actLayer(kind=out_act)

    def forward(self,style, en_feats,noise=None):
        out = {"noise": [], "freq": [], "img": None,
               "en_feats": en_feats}
        noise = NoiseManager(noise, self.device, self.trace_model)

        hidden = en_feats[0]

        for i, m in enumerate(self.layers):
            start_style_idx = m.wsize()*i + 1
            end_style_idx = m.wsize()*i + m.wsize() + 1
            out["noise"].append(noise(2 * hidden.size(-1), 2))
            hidden, freq = m(hidden,style if len(style.shape)==2 else style[:, start_style_idx:end_style_idx, :],
                             en_feats[i+1],
                             noise=out["noise"][-1])

            out["freq"].append(freq)

        out["img"] = self.dwt_to_img(out["freq"][-1])

        out["img"] = self.out_act(out["img"])

        return out

    def dwt_to_img(self, img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return self.idwt((low, [high]))

    def wsize(self):
        return len(self.layers) * self.layers[0].wsize() + 2

class MobileSynthesisNetwork_v5(nn.Module):
    def __init__(
            self,
            style_dim,
            channels = [512, 512, 512, 512, 256, 128, 64],
            device = 'cuda',
            out_act = 'tanh',
            trace_model = False,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.device = device

        self.layers = nn.ModuleList()
        channels_in = channels[0]
        for i, channels_out in enumerate(channels[1:]):
            self.layers.append(
                MobileSynthesisBlock_v5(
                    channels_in,
                    channels_out,
                    style_dim,
                    3,
                    conv_module = ModulatedDWConv2d,
                    # use_spatial_att = False if i >=2 else True,
                    use_spatial_att = False
                )
            )
            channels_in = channels_out

        self.trace_model = trace_model
        self.idwt = DWTInverse(mode="zero", wave="db1",trace_model=self.trace_model)
        self.out_act = actLayer(kind=out_act)

    def forward(self,style, en_feats,noise=None):
        out = {"noise": [], "freq": [], "img": None,
               "en_feats": en_feats}
        noise = NoiseManager(noise, self.device, self.trace_model)

        hidden = en_feats[0]

        for i, m in enumerate(self.layers):
            start_style_idx = m.wsize()*i + 1
            end_style_idx = m.wsize()*i + m.wsize() + 1
            out["noise"].append(noise(2 * hidden.size(-1), 2))
            hidden, freq = m(hidden,style if len(style.shape)==2 else style[:, start_style_idx:end_style_idx, :],
                             en_feats[i+1],
                             noise=out["noise"][-1])

            out["freq"].append(freq)

        out["img"] = self.dwt_to_img(out["freq"][-1])

        out["img"] = self.out_act(out["img"])

        return out

    def dwt_to_img(self, img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return self.idwt((low, [high]))

    def wsize(self):
        return len(self.layers) * self.layers[0].wsize() + 2

if __name__ == '__main__':
    x = torch.randn(1,256).cuda()
    # model = StyleEncoder(256).to('cuda')
    model = MobileSynthesisNetwork(style_dim=512)
    print_network_params(model,"model")
    summary(model,(256,))
