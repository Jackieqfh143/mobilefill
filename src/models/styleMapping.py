from src.modules.legacy import StyledConv
from src.modules.constant_input import ConstantInput
from src.modules.mobile_synthesis_block import MobileSynthesisBlock
from src.modules.mobile_synthesis_block import MobileSynthesisBlock,MobileSynthesisBlock_withFFC,\
    MobileSynthesisBlock_v2,MobileSynthesisBlock_v3,MobileSynthesisBlock_v4
from src.modules.modulated_conv2d import ModulatedDWConv2d
from src.modules.legacy import EqualLinear
from src.modules.ffc import actLayer
from src.modules.idwt_upsample import DWTInverse
from src.utils.noise_manager import NoiseManager
import torch
import torch.nn as nn


class StyleEncoder(nn.Module):
    def __init__(self,
                 latent_nc = 512,
                 channels=[512, 512, 512],
                 blur_kernel=[1, 3, 3, 1],
                 device='cuda',
                 ):
        super(StyleEncoder, self).__init__()
        style_dim = latent_nc // 2
        self.ws_style = EqualLinear(latent_nc, style_dim, activation="fused_lrelu")
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
        style = self.ws_style(style)
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

        return hidden,style,freq

class StyleEncoder_v2(nn.Module):
    def __init__(self,
                 latent_nc = 512,
                 channels=[512, 512, 512],
                 blur_kernel=[1, 3, 3, 1],
                 device='cuda',
                 ):
        super(StyleEncoder_v2, self).__init__()
        style_dim = int(latent_nc * 0.75)
        self.ws_style = EqualLinear(latent_nc, style_dim, activation="fused_lrelu")
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
        style = self.ws_style(style)
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

        return hidden,style,freq

class StyleEncoder_v3(nn.Module):
    def __init__(self,
                 latent_nc = 512,
                 channels=[512, 512, 512],
                 blur_kernel=[1, 3, 3, 1],
                 device='cuda',
                 ):
        super(StyleEncoder_v3, self).__init__()
        # style_dim = int(latent_nc * 0.75)
        # self.ws_style = EqualLinear(latent_nc, style_dim, activation="fused_lrelu")
        self.device = device
        self.input = ConstantInput(channels[0])
        self.conv1 = StyledConv(
            channels[0], channels[0], 3, latent_nc, blur_kernel=blur_kernel
        )
        self.layers = nn.ModuleList()
        channels_in = channels[0]
        for i, channels_out in enumerate(channels[1:]):
            self.layers.append(
                MobileSynthesisBlock(
                    channels_in,
                    channels_out,
                    latent_nc,
                    3,
                    conv_module=ModulatedDWConv2d,
                )
            )
            channels_in = channels_out

    def forward(self, style):
        # style = self.ws_style(style)
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

        return hidden,style,freq

class StyleEncoder_v4(nn.Module):
    def __init__(self,
                 latent_nc = 512,
                 channels=[256, 512, 512],
                 blur_kernel=[1, 3, 3, 1],
                 device='cuda',
                 ):
        super(StyleEncoder_v4, self).__init__()
        self.device = device
        self.input = ConstantInput(channels[0])
        self.conv1 = StyledConv(
            channels[0], channels[0], 3, latent_nc, blur_kernel=blur_kernel
        )
        self.layers = nn.ModuleList()
        channels_in = channels[0]
        for i, channels_out in enumerate(channels[1:]):
            self.layers.append(
                MobileSynthesisBlock(
                    channels_in,
                    channels_out,
                    latent_nc,
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

        return hidden,style,freq

class StyleEncoder_v5(nn.Module):
    def __init__(self,
                 latent_nc = 512,
                 channels=[512, 512, 512, 512, 256, 128, 64],
                 blur_kernel=[1, 3, 3, 1],
                 device='cuda',
                 trace_model = False
                 ):
        super(StyleEncoder_v5, self).__init__()
        self.device = device
        self.input = ConstantInput(channels[0])
        self.conv1 = StyledConv(
            channels[0], channels[0], 3, latent_nc, blur_kernel=blur_kernel
        )
        self.layers = nn.ModuleList()
        channels_in = channels[0]
        for i, channels_out in enumerate(channels[1:]):
            self.layers.append(
                MobileSynthesisBlock(
                    channels_in,
                    channels_out,
                    latent_nc,
                    3,
                    conv_module=ModulatedDWConv2d,
                )
            )
            channels_in = channels_out

        self.trace_model = trace_model
        self.idwt = DWTInverse(mode="zero", wave="db1", trace_model=self.trace_model)
        self.out_act = actLayer(kind='tanh')

    def forward(self, style, en_feats, noise = None):
        out = {"noise": [], "freq": [], "img": None}
        hidden = self.input(style[:,0])
        _noise = torch.randn(1, 1, hidden.size(-2), hidden.size(-1)).to(style.device)
        hidden = self.conv1(hidden, style[:,1], noise=_noise)

        noise = NoiseManager(noise, self.device, self.trace_model)

        for i, m in enumerate(self.layers):
            start_style_idx = m.wsize() * i + 1
            end_style_idx = m.wsize() * i + m.wsize() + 1
            out["noise"].append(noise(2 * hidden.size(-1), 2))
            hidden, freq = m(hidden, style if len(style.shape) == 2 else style[:, start_style_idx:end_style_idx, :],
                             noise=out["noise"][-1])
            out["freq"].append(freq)

        out["img"] = self.dwt_to_img(out["freq"][-1])

        out["img"] = self.out_act(out["img"])
        return out['img']

    def dwt_to_img(self, img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return self.idwt((low, [high]))

    def wsize(self):
        return len(self.layers) * self.layers[0].wsize() + 2

class StyleEncoder_v6(nn.Module):
    def __init__(self,
                 latent_nc = 512,
                 channels=[512, 512, 512, 512, 256, 128, 64],
                 blur_kernel=[1, 3, 3, 1],
                 device='cuda',
                 trace_model = False
                 ):
        super(StyleEncoder_v6, self).__init__()
        self.device = device
        self.input = ConstantInput(channels[0])
        self.conv1 = StyledConv(
            channels[0], channels[0], 3, latent_nc, blur_kernel=blur_kernel
        )
        self.layers = nn.ModuleList()
        channels_in = channels[0]
        for i, channels_out in enumerate(channels[1:]):
            self.layers.append(
                MobileSynthesisBlock_v4(
                    channels_in,
                    channels_out,
                    latent_nc,
                    3,
                    conv_module=ModulatedDWConv2d,
                )
            )
            channels_in = channels_out

        self.trace_model = trace_model
        self.idwt = DWTInverse(mode="zero", wave="db1", trace_model=self.trace_model)
        self.out_act = actLayer(kind='tanh')

    def forward(self, style, feats, noise = None):
        out = {"noise": [], "feats":feats, "freq": [], "img": None}
        hidden = self.input(style[:,0])
        _noise = torch.randn(1, 1, hidden.size(-2), hidden.size(-1)).to(style.device)
        hidden = self.conv1(hidden, style[:,1], noise=_noise)

        noise = NoiseManager(noise, self.device, self.trace_model)

        for i, m in enumerate(self.layers):
            start_style_idx = m.wsize() * i + 1
            end_style_idx = m.wsize() * i + m.wsize() + 1
            out["noise"].append(noise(2 * hidden.size(-1), 2))
            hidden, freq = m(hidden, style if len(style.shape) == 2 else style[:, start_style_idx:end_style_idx, :],feats,
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