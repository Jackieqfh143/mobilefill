import traceback
import torch
import torch.nn as nn
import torchvision
from src.models.mobileFill import MobileFill


def apply_trace_model_mode(mode=False):
    def _apply_trace_model_mode(m):
        if hasattr(m, 'trace_model'):
            m.trace_model = mode
    return _apply_trace_model_mode


class Wrapper(nn.Module):
    def __init__(
            self,
            synthesis_network,
            style_tmp,
            en_feats_tmp,
    ):
        super().__init__()
        self.m = synthesis_network
        self.noise = self.m(style_tmp,en_feats_tmp)["noise"]

    def forward(self, style,en_feats):
        return self.m(style,en_feats,noise=self.noise)["img"]



if __name__ == '__main__':
    dummy_input = torch.randn(1, 4, 256, 256, device="cpu")
    mask = torch.randn(1, 1, 256, 256,device="cpu")
    # model_path = '/home/codeoops/CV/MobileFill/checkpoints/G-step=325500_lr=0.0001_ema_loss=0.4704.pth'
    device = 'cpu'
    model = MobileFill(device=device, input_nc=4,target_size=256)
    model.eval().requires_grad_(False)
    # model.load_state_dict(torch.load(model_path,map_location='cpu'))

    style = torch.randn(1, 512).to(device)
    model.encoder.apply(apply_trace_model_mode(True))
    torch.onnx.export(
        model.encoder,
        (dummy_input,style),
        "./encoder.onnx",
        input_names=['input','styles'],
        output_names=["co_styles", "en_feats"],
        verbose=False,
        dynamic_axes={
            "input":[0],
            "styles":[0],
            "en_feats":[0],
            "co_styles": [0],
        }
    )

    noise = torch.randn(1, 512).to(device)

    model.mapping_net.apply(apply_trace_model_mode(True))
    torch.onnx.export(
        model.mapping_net,
        (noise, ),
        "./mappingNet.onnx",
        input_names=['noise'],
        output_names=['style'],
        verbose=False,
        dynamic_axes={
            'noise':[0],
            'style':[0]
        }
    )

    en_feats = [torch.randn(1, 512, 16, 16,device="cpu"),
                torch.randn(1, 512, 32, 32,device="cpu"),
                torch.randn(1, 256, 64, 64,device="cpu"),
                torch.randn(1, 128, 128, 128,device="cpu"),
                ]

    style = torch.randn(1, 512).to(device)
    style = style.unsqueeze(1).repeat(1, 11, 1)

    model.generator.apply(apply_trace_model_mode(True))

    torch.onnx.export(
        Wrapper(model.generator, style, en_feats),
        (style, en_feats,),
        "./generator.onnx",
        input_names=['style', 'en_feats'],
        output_names=['out'],
        verbose=False,
        dynamic_axes={
            "style": [0],
            "en_feats": [0],
            'out': [0]
        }
    )











