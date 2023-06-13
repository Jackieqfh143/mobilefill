from src.modules.ffc import actLayer,upsample,SeperableConv
from src.models.encoder import EESP_V4
from src.modules.attention import ECAAttention
from src.modules.MobileViT import MobileViT, MobileViT_V2, MobileViT_V3
from torch.nn import init
from complexity import *


class LatentDecoder(nn.Module):
    def __init__(self, config = [512, 256, 256, 128], input_size = 16, out_dim = 3, up_blocks = 4):
        super(LatentDecoder, self).__init__()
        # self.transformer = MobileViT_V3(in_channels=config[0], image_size=(input_size, input_size))
        self.up_blocks = up_blocks
        self.layers = nn.ModuleList()
        for i in range(up_blocks):
            idx = min(len(config) - 1, i + 1)
            self.layers.append(DecoderBlock(in_dim=config[i], out_dim=config[idx]))

        self.out_affine = nn.Conv2d(in_channels=config[-1], out_channels=out_dim, kernel_size=3, padding=1)
        self.out_act = actLayer(kind='tanh')

    def forward(self, x, en_feats):
        # hidden = self.transformer(x)
        hidden = x
        for i, module in enumerate(self.layers):
            hidden = module(hidden, en_feats[i])
        out_x = self.out_affine(hidden)
        return self.out_act(out_x)


class Decoder(nn.Module):
    def __init__(self ,in_dim, out_dim = 3, up_blocks = 4):
        super(Decoder, self).__init__()
        self.up_blocks = up_blocks
        dim = in_dim
        self.layers = nn.ModuleList()
        for i in range(up_blocks):
            self.layers.append(DecoderBlock(in_dim = dim, out_dim = dim // 2))
            dim = dim // 2

        self.out_affine = nn.Conv2d(in_channels=dim, out_channels = out_dim, kernel_size=3, padding=1)
        self.out_act = actLayer(kind='tanh')

    def forward(self, x, en_feats):
        hidden = x
        for i, module in enumerate(self.layers):
            hidden = module(hidden,en_feats[i])
        out_x = self.out_affine(hidden)
        return self.out_act(out_x)


class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DecoderBlock, self).__init__()
        eesp_res_blocks = []
        for i in range(3):
            eesp_res_blocks.append(EESP_V4(in_dim, in_dim, stride=1, d_rates=[1, 2, 4]))
        self.eesp_res_block = nn.Sequential(*eesp_res_blocks)
        self.gate = nn.Sequential(
            SeperableConv(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.upsample_block = upsample(scale_factor=2, kind="interpolate")

        self.upsample_skip = nn.Sequential(upsample(scale_factor=2, kind="shuffle"),
                                           ECAAttention(),
                                           nn.Conv2d(in_channels=in_dim // 4, out_channels=in_dim, kernel_size=1))
        self.out_affine = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)

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

    def forward(self, hidden, en_feat):
        skip = self.upsample_skip(hidden)
        mask = self.gate(en_feat)
        hidden = hidden + mask * en_feat
        hidden = self.upsample_block(hidden)
        hidden = self.eesp_res_block(hidden)
        hidden = hidden + skip
        return self.out_affine(hidden)


if __name__ == '__main__':
    input = torch.randn(1, 512, 16, 16)
    en_feats = [
        torch.randn(1, 512, 16, 16),
        torch.randn(1, 256, 32, 32),
        torch.randn(1, 256, 64, 64),
        torch.randn(1, 128, 128, 128)
    ]
    model = LatentDecoder(config=[512, 256, 256, 128], out_dim = 3,up_blocks= 4)
    print_network_params(model, "model")
    flop_counter(model, (input, en_feats))
    out = model(input,en_feats)


