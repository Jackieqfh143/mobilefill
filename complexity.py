from torchscan.crawler import crawl_module
from fvcore.nn import FlopCountAnalysis
import torch.nn as nn
import torch


def parse_shapes(input):
    if isinstance(input, list) or isinstance(input,tuple):
        out_shapes = [item.shape[1:] for item in input]
    else:
        out_shapes = input.shape[1:]

    return out_shapes

def flop_counter(model,input):
    try:
        module_info = crawl_module(model, parse_shapes(input))
        flops = sum(layer["flops"] for layer in module_info["layers"])
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
            print(f'FLOPs : {flops:.5f}')
            return flops

    else:
        flops = flops / 1e9
        print(f'FLOPs : {flops:.5f}')
        return flops

def print_network_params(model,model_name):
    num_params = 0
    if isinstance(model,list):
        for m in model:
            for param in m.parameters():
                num_params += param.numel()
        print('[Network %s] Total number of parameters : %.2f M' % (model_name, num_params / 1e6))

    else:
        for param in model.parameters():
            num_params += param.numel()
        print('[Network %s] Total number of parameters : %.2f M' % (model_name, num_params / 1e6))



