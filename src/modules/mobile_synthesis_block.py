import torch

from .styled_conv2d import *
from .multichannel_image import *
from .modulated_conv2d import *
from .idwt_upsample import *
from .ffc import *
import torch.nn.functional as F
from src.modules.attention import CBAMBlock
from src.models.encoder import EESP,EESP_V2,EESP_V3,EESP_V4
from src.models.discriminator import Self_Attn,AttentionModule,MobileAttentionModule,AttentionModule_V2
from src.modules.attention import ECAAttention
from src.utils.diffaug import rand_cutout
import numpy as np



class MobileSynthesisBlock(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size=3,
            conv_module=ModulatedConv2d
    ):
        super().__init__()
        self.up = IDWTUpsaplme(channels_in, style_dim)
        self.conv1 = StyledConv2d(
            channels_in // 4,
            channels_out,
            style_dim,
            kernel_size,
            conv_module=conv_module
        )
        self.conv2 = StyledConv2d(
            channels_out,
            channels_out,
            style_dim,
            kernel_size,
            conv_module=conv_module
        )
        self.to_img = MultichannelIamge(
            channels_in=channels_out,
            channels_out=12,
            style_dim=style_dim,
            kernel_size=1
        )

    def forward(self, hidden, style, noise=[None, None]):
        hidden = self.up(hidden, style if style.ndim == 2 else style[:, 0, :])
        hidden = self.conv1(hidden, style if style.ndim == 2 else style[:, 0, :], noise=noise[0])
        hidden = self.conv2(hidden, style if style.ndim == 2 else style[:, 1, :], noise=noise[1])
        img = self.to_img(hidden, style if style.ndim == 2 else style[:, 2, :])
        return hidden, img

    def wsize(self):
        return 3

class MobileSynthesisBlock_withFFC(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size=3,
            conv_module=ModulatedDWConv2d,
            use_spatial_att = True
    ):
        super().__init__()
        self.up = IDWTUpsaplme(channels_in, style_dim)
        self.conv1 = StyledConv2d(
            channels_in // 4,
            channels_out,
            style_dim,
            kernel_size,
            conv_module=conv_module
        )
        self.conv2 = StyledConv2d(
            channels_out,
            channels_out,
            style_dim,
            kernel_size,
            conv_module=conv_module
        )

        # eesp_res_blocks = []
        # for i in range(4):
        #     eesp_res_blocks.append(EESP_V2(channels_out, channels_out, stride=1, d_rates=[1, 2, 4, 8]))
        # self.eesp_res_block_en = nn.Sequential(*eesp_res_blocks)

        eesp_res_blocks = []
        for i in range(3):
            eesp_res_blocks.append(EESP_V4(channels_out,channels_out, stride=1, d_rates=[1,2,4]))
        self.eesp_res_block = nn.Sequential(*eesp_res_blocks)

        eesp_res_blocks = []
        for i in range(3):
            eesp_res_blocks.append(EESP_V4(channels_out, channels_out, stride=1, d_rates=[1,2,4]))
        self.eesp_res_block_merge = nn.Sequential(*eesp_res_blocks)

        self.gate = nn.Sequential(
            SeperableConv(in_channels=channels_out, out_channels=channels_out, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # self.att = AttentionModule(in_nc=channels_out,use_spatial_att=use_spatial_att)

        self.att = ECAAttention()
        # self.att = AttentionModule(in_nc=channels_out, use_spatial_att=use_spatial_att)

        self.to_img = MultichannelIamge(
            channels_in=channels_out,
            channels_out=12,
            style_dim=style_dim,
            kernel_size=1
        )



    def forward(self, hidden, style, en_feat,noise=[None, None]):
        hidden = self.up(hidden, style if style.ndim == 2 else style[:, 0, :])
        mul_map = torch.ones_like(hidden) * 0.5
        mul_map = F.dropout(mul_map, training=True)
        hidden = mul_map * hidden
        # en_feat = en_feats.get(str(hidden.size(-1)),None) #find en_feat by resolution
        hidden = self.conv1(hidden, style if style.ndim == 2 else style[:, 0, :], noise=noise[0])
        hidden = self.eesp_res_block(hidden)

        # if en_feat != None:
        #     hidden_res = hidden
        #     en_feat = self.eesp_res_block_en(en_feat)
        #     hidden_z = hidden + en_feat
        #     if self.use_ffc:
        #         # hidden_z = self.ffc_res_block(hidden_z)
        #         hidden_z = self.eesp_res_block(hidden_z)
        #         # hidden_z = self.att(hidden_z)
        #         # hidden_z = hidden_res + hidden_z
        #         # hidden_z = hidden_res + hidden_z
        #     # mul_map = torch.ones_like(hidden_z) * 0.5
        #     # mul_map = F.dropout(mul_map, training=True)
        #     # hidden = hidden_z * mul_map + hidden * (1 - mul_map)
        #     hidden = hidden_res + hidden_z

        hidden = self.conv2(hidden, style if style.ndim == 2 else style[:, 1, :], noise=noise[1])

        gamma = self.gate(en_feat)
        # en_feat = self.eesp_res_block_en(en_feat)
        hidden = en_feat * gamma + hidden * (1 - gamma)
        hidden = self.att(hidden)

        hidden = self.eesp_res_block_merge(hidden)

        img = self.to_img(hidden, style if style.ndim == 2 else style[:, 2, :])
        return hidden, img

    def wsize(self):
        return 3

class MobileSynthesisBlock_v2(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size=3,
            conv_module=ModulatedDWConv2d,
            use_spatial_att = True
    ):
        super().__init__()
        self.up = IDWTUpsaplme(channels_in, style_dim)
        self.conv1 = StyledConv2d(
            channels_in // 4,
            channels_out,
            style_dim,
            kernel_size,
            conv_module=conv_module
        )
        self.conv2 = StyledConv2d(
            channels_out,
            channels_out,
            style_dim,
            kernel_size,
            conv_module=conv_module
        )

        eesp_res_blocks = []
        for i in range(3):
            eesp_res_blocks.append(EESP_V4(channels_out,channels_out, stride=1, d_rates=[1,2,4]))
        self.eesp_res_block = nn.Sequential(*eesp_res_blocks)

        eesp_res_blocks = []
        for i in range(3):
            eesp_res_blocks.append(EESP_V4(channels_out, channels_out, stride=1, d_rates=[1,2,4]))
        self.eesp_res_block_merge = nn.Sequential(*eesp_res_blocks)

        self.gate = nn.Sequential(
            SeperableConv(in_channels=channels_out, out_channels=channels_out, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.att = ECAAttention()

        self.to_img = MultichannelIamge(
            channels_in=channels_out,
            channels_out=12,
            style_dim=style_dim,
            kernel_size=1
        )



    def forward(self, hidden, style, en_feat,noise=[None, None]):
        hidden = self.up(hidden, style if style.ndim == 2 else style[:, 0, :])
        hidden = self.conv1(hidden, style if style.ndim == 2 else style[:, 0, :], noise=noise[0])
        hidden = self.conv2(hidden, style if style.ndim == 2 else style[:, 1, :], noise=noise[1])

        mul_map = torch.ones_like(hidden) * 0.5
        mul_map = F.dropout(mul_map, training=True)
        hidden = mul_map * hidden
        hidden = self.eesp_res_block(hidden)

        gamma = self.gate(en_feat)
        hidden = en_feat * gamma + hidden * (1 - gamma)
        hidden = self.att(hidden)

        hidden = self.eesp_res_block_merge(hidden)

        img = self.to_img(hidden, style if style.ndim == 2 else style[:, 2, :])
        return hidden, img

    def wsize(self):
        return 3

class MobileSynthesisBlock_v3(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size=3,
            conv_module=ModulatedDWConv2d,
            use_spatial_att = True
    ):
        super().__init__()
        self.up = IDWTUpsaplme(channels_in, style_dim)
        self.conv1 = StyledConv2d(
            channels_in // 4,
            channels_out,
            style_dim,
            kernel_size,
            conv_module=conv_module
        )
        self.conv2 = StyledConv2d(
            channels_out,
            channels_out,
            style_dim,
            kernel_size,
            conv_module=conv_module
        )

        eesp_res_blocks = []
        for i in range(3):
            eesp_res_blocks.append(EESP_V4(channels_out,channels_out, stride=1, d_rates=[1,2,4]))
        self.eesp_res_block = nn.Sequential(*eesp_res_blocks)

        eesp_res_blocks = []
        for i in range(3):
            eesp_res_blocks.append(EESP_V4(channels_out, channels_out, stride=1, d_rates=[1,2,4]))
        self.eesp_res_block_merge = nn.Sequential(*eesp_res_blocks)

        self.gate = nn.Sequential(
            SeperableConv(in_channels=channels_out, out_channels=channels_out, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.att = ECAAttention()

        self.to_img = MultichannelIamge(
            channels_in=channels_out,
            channels_out=12,
            style_dim=style_dim,
            kernel_size=1
        )



    def forward(self, hidden, style, en_feat,noise=[None, None]):
        hidden = self.up(hidden, style if style.ndim == 2 else style[:, 0, :])
        hidden = self.conv1(hidden, style if style.ndim == 2 else style[:, 0, :], noise=noise[0])
        hidden = self.eesp_res_block(hidden)
        hidden = self.conv2(hidden, style if style.ndim == 2 else style[:, 1, :], noise=noise[1])

        # mul_map = torch.ones_like(hidden) * 0.5
        # mul_map = F.dropout(mul_map, training=True)
        # hidden = mul_map * hidden

        gamma = self.gate(en_feat)
        hidden = en_feat * gamma + hidden * (1 - gamma)
        hidden = self.att(hidden)

        hidden = self.eesp_res_block_merge(hidden)

        img = self.to_img(hidden, style if style.ndim == 2 else style[:, 2, :])
        return hidden, img

    def wsize(self):
        return 3

class MobileSynthesisBlock_v4(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size=3,
            conv_module=ModulatedDWConv2d,
            use_spatial_att = True
    ):
        super().__init__()
        self.up = IDWTUpsaplme(channels_in, style_dim)
        self.conv1 = StyledConv2d(
            channels_in // 4,
            channels_out,
            style_dim,
            kernel_size,
            conv_module=conv_module
        )
        self.conv2 = StyledConv2d(
            channels_out,
            channels_out,
            style_dim,
            kernel_size,
            conv_module=conv_module
        )

        eesp_res_blocks = []
        for i in range(3):
            eesp_res_blocks.append(EESP_V4(channels_out,channels_out, stride=1, d_rates=[1,2,4]))
        self.eesp_res_block = nn.Sequential(*eesp_res_blocks)

        eesp_res_blocks = []
        for i in range(3):
            eesp_res_blocks.append(EESP_V4(channels_out, channels_out, stride=1, d_rates=[1,2,4]))
        self.eesp_res_block_merge = nn.Sequential(*eesp_res_blocks)

        self.gate = nn.Sequential(
            SeperableConv(in_channels=channels_out, out_channels=channels_out, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.att = ECAAttention()

        self.to_img = MultichannelIamge(
            channels_in=channels_out,
            channels_out=12,
            style_dim=style_dim,
            kernel_size=1
        )



    def forward(self, hidden, style, en_feat,noise=[None, None]):
        hidden = self.up(hidden, style if style.ndim == 2 else style[:, 0, :])
        hidden = self.conv1(hidden, style if style.ndim == 2 else style[:, 0, :], noise=noise[0])
        hidden = self.eesp_res_block(hidden)
        hidden = self.conv2(hidden, style if style.ndim == 2 else style[:, 1, :], noise=noise[1])

        if en_feat.get(hidden.size(-1)) != None:
            feat = en_feat.get(hidden.size(-1))
            gamma = self.gate(feat)
            hidden = feat * gamma + hidden * (1 - gamma)

        hidden = self.att(hidden)

        hidden = self.eesp_res_block_merge(hidden)

        img = self.to_img(hidden, style if style.ndim == 2 else style[:, 2, :])
        return hidden, img

    def wsize(self):
        return 3

class MobileSynthesisBlock_v5(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size=3,
            conv_module=ModulatedDWConv2d,
            use_spatial_att = True
    ):
        super().__init__()
        self.up = IDWTUpsaplme(channels_in, style_dim)
        self.conv1 = StyledConv2d(
            channels_in // 4,
            channels_out,
            style_dim,
            kernel_size,
            conv_module=conv_module
        )
        self.conv2 = StyledConv2d(
            channels_out,
            channels_out,
            style_dim,
            kernel_size,
            conv_module=conv_module
        )

        eesp_res_blocks = []
        for i in range(3):
            eesp_res_blocks.append(EESP_V4(channels_out,channels_out, stride=1, d_rates=[1,2,4]))
        self.eesp_res_block = nn.Sequential(*eesp_res_blocks)
        #
        # eesp_res_blocks = []
        # for i in range(3):
        #     eesp_res_blocks.append(EESP_V4(channels_out, channels_out, stride=1, d_rates=[1,2,4]))
        # self.eesp_res_block_merge = nn.Sequential(*eesp_res_blocks)
        #
        # self.gate = nn.Sequential(
        #     SeperableConv(in_channels=channels_out, out_channels=channels_out, kernel_size=3, padding=1),
        #     nn.Sigmoid()
        # )
        #
        self.att = ECAAttention()

        self.to_img = MultichannelIamge(
            channels_in=channels_out,
            channels_out=12,
            style_dim=style_dim,
            kernel_size=1
        )



    def forward(self, hidden, style, en_feat,noise=[None, None]):
        hidden = self.up(hidden, style if style.ndim == 2 else style[:, 0, :])
        hidden = self.conv1(hidden, style if style.ndim == 2 else style[:, 0, :], noise=noise[0])
        # mask = rand_cutout(en_feat, ratio = 0.5 * np.random.rand(1)[0])
        hidden = hidden + en_feat
        hidden = self.eesp_res_block(hidden)
        hidden = self.att(hidden)
        hidden = self.conv2(hidden, style if style.ndim == 2 else style[:, 1, :], noise=noise[1])

        img = self.to_img(hidden, style if style.ndim == 2 else style[:, 2, :])
        return hidden, img

    def wsize(self):
        return 3
