# from torchscan.crawler import crawl_module
from fvcore.nn import FlopCountAnalysis
import torch.nn as nn
import torch


def parse_shapes(input):
    if isinstance(input, list) or isinstance(input,tuple):
        out_shapes = [item.shape[1:] for item in input]
    elif isinstance(input,dict):
        out_shapes = [item.shape[1:] for item in input.values()]
    else:
        out_shapes = input.shape[1:]

    return out_shapes

def flop_counter(model,input):
    try:
        raise  Exception
        # module_info = crawl_module(model, parse_shapes(input))
        # flops = sum(layer["flops"] for layer in module_info["layers"])
        # if flops == 0.0:
        #     raise
    except Exception as e:
        print(f'\nflops counter came across error: {e} \n')
        try:
            print('try another counter...\n')
            if isinstance(input, list):
                input = tuple(input)
            flops = FlopCountAnalysis(model, input).total()
        except Exception as e:
            print(e)
            raise e
        else:
            flops = flops / 1e9
            print(f'FLOPs : {flops:.5f} G')
            return flops

    else:
        flops = flops / 1e9
        print(f'FLOPs : {flops:.5f} G')
        return flops

def print_network_params(model,model_name):
    num_params = 0
    if isinstance(model,list):
        for m in model:
            for param in m.parameters():
                num_params += param.numel()
        print('[Network %s] Total number of parameters : %.5f M' % (model_name, num_params / 1e6))

    else:
        for param in model.parameters():
            num_params += param.numel()
        print('[Network %s] Total number of parameters : %.5f M' % (model_name, num_params / 1e6))



#SpatialGroupEnhance
class SGE(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)

        return x




if __name__ == '__main__':
    x = torch.randn(1,256,32,32)
    model = SGE(groups=4)
    out = model(x)
    print_network_params(model,'SGE')
    flop_counter(model,x)   #support multiple input

