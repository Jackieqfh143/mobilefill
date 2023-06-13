import math

from src.models.generator import MobileSynthesisNetwork,MobileSynthesisNetwork_v2,\
    MobileSynthesisNetwork_v3,MobileSynthesisNetwork_v4,MobileSynthesisNetwork_v5, MobileSynthesisNetwork_v6
from src.models.mapping_network import MappingNetwork
from src.models.encoder import EESPNet,LFFC_encoder,EESPNet_v10,EESPNet_v11,EESPNet_v12,EESPNet_v13,\
    EESPNet_v14,EESPNet_v15,EESPNet_v16,EESPNet_v17,EESPNet_v18,EESPNet_v19,EESPNet_v20,EESPNet_v21,EESPNet_v22
from src.models.styleMapping import StyleEncoder,StyleEncoder_v2,StyleEncoder_v3,StyleEncoder_v4,\
    StyleEncoder_v5,StyleEncoder_v6
from src.modules.legacy import EqualLinear
from pytorch_wavelets import DWTInverse, DWTForward
import torch.nn as nn
import numpy as np
from PIL import Image
from src.utils.util import tensor2cv,cv2tensor
import torch
import random

channels_config = {512: {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 64,
        }, 256: {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 64,
        }}


class MobileFill(nn.Module):
    def __init__(self,device= 'cuda',target_size=512, input_nc=4,down_num = 4, latent_nc=512,mlp_layers=8):
        super(MobileFill, self).__init__()
        # self.channels = {
        #     4: 256,
        #     8: 512,
        #     16: 512,
        #     32: 512,
        #     64: 256,
        #     128: 128,
        #     256: 64,
        #     512: 64,
        # }

        # down_num = 4 if target_size == 256 else 5
        # en_channels = [v for k, v in self.channels.items() if k < target_size][::-1]
        # gen_channels = [v for k,v in self.channels.items() if k >= (target_size // 2**down_num) and (k <= target_size //2)]

        en_channels = [128, 256, 256, 512]
        gen_channels = [512, 256, 256, 128]
        # gen_channels = [v for k, v in self.channels.items() if k < target_size]
        self.device = device
        # self.encoder = EESPNet_v2(input_nc=input_nc,output_nc=latent_nc , input_size=target_size).to(device)
        self.encoder = EESPNet_v22(config=en_channels, input_nc=input_nc, output_nc=latent_nc, input_size=target_size, down_num=down_num).to(device)
        # self.encoder = LFFC_encoder(input_nc=input_nc,latent_nc=latent_nc,ngf=64,n_downsampling=4,n_blocks=4).to(device)
        # self.encoder = mobilevit_xs(num_classes=latent_nc).to(device)
        self.mapping_net = MappingNetwork(style_dim=latent_nc,n_layers=mlp_layers).to(device)
        # self.generator = StyleEncoder_v6(channels= gen_channels, latent_nc = latent_nc, device=device).to(device)
        self.generator = MobileSynthesisNetwork_v6(style_dim=latent_nc,channels=gen_channels,device=device).to(device)
        # self.co_mod_layer = nn.Linear(in_features=2 * latent_nc, out_features= latent_nc).to(self.device)
        self.latent_nc = latent_nc
        self.latent_num = self.generator.wsize()
        self.dwt = DWTForward(J=1, mode='zero', wave='db1').to(self.device)
        self.idwt = DWTInverse(mode="zero", wave="db1").to(self.device)


    def preprocess(self,imgs,masks):
        if not isinstance(imgs,list):
            imgs = [imgs]
            masks = [masks]

        for i in range(len(imgs)):
            if isinstance(imgs[i],str):
                imgs[i] = np.array(Image.open(imgs[i]))
                masks[i] = np.array(Image.open(masks[i]))

            if len(masks[i].shape) < 3:
                masks[i] = np.expand_dims(masks[i],axis=-1)

        imgs_t = cv2tensor(imgs).to(self.device)
        masks_t = cv2tensor(masks).to(self.device)

        if masks_t.size(1) > 1:
            masks_t = masks_t[:, 2:3, :, :]

        self.masks_t = 1 - masks_t      #0 for holes
        self.imgs_t = imgs_t / 0.5 - 1.0

        # self.imgs_t = self.img_to_dwt(self.imgs_t)
        self.masked_imgs = self.imgs_t * self.masks_t
        input_imgs = torch.cat((self.masked_imgs, self.masks_t), dim=1)
        return input_imgs

    def postprocess(self,imgs_t,to_cv=True):
        imgs_t = (imgs_t + 1.0) * 0.5  #scale to 0 ~ 1
        if not to_cv:
            return imgs_t

        return tensor2cv(imgs_t)

    @torch.no_grad()
    def compute_mean_style(self, style_dim, wsize=1, batch_size=4096):
        style = self.mapping_net(torch.randn(batch_size, style_dim)).mean(0, keepdim=True)
        if wsize != 1:
            style = style.unsqueeze(1).repeat(1, wsize, 1)
        return style

    def mixing_style(self, batch_size, n_images = 4096):
        noise1 = torch.randn(batch_size, self.latent_nc).to(self.device)
        style1 = self.mapping_net(noise1)

        noise2 = torch.randn(batch_size, self.latent_nc).to(self.device)
        style2 = self.mapping_net(noise2)

        diff = (style2 - style1) / n_images

        return style1 + diff * random.randint(0, n_images)

    def latent_augmented_sampling(self, batch_size, radius = 0.001, num_negative = 10):
        query = self.make_style(batch_size)
        pos = torch.FloatTensor(query.shape).uniform_(-radius,radius).add_(query)
        negs = []
        for k in range(num_negative):
            neg = self.make_style(batch_size)
            while (neg-query).abs().min() < self.opt.radius:
                neg = self.make_style(batch_size)
            negs.append(neg)
        return query, pos, negs

    def make_style(self,batch_size):
        noise = torch.randn(batch_size, self.latent_nc).to(self.device)
        style = self.mapping_net(noise)
        return style

    def mode_seeking_loss(self, img1, img2, ws1, ws2):
        eps = 1 * 1e-5
        loss1 = torch.mean(torch.abs(img1["img"] - img2["img"])) / torch.mean(torch.abs(ws1 - ws2))
        loss2 = torch.mean(torch.abs(img1["freq"][-1] - img2["freq"][-1])) / torch.mean(torch.abs(ws1 - ws2))
        loss_lz = 1 / (loss1 + + loss2 + eps)
        return loss_lz

    def get_scores(self,img1,img2,ws1,ws2):
        eps = 1 * 1e-5
        loss1 = torch.mean(torch.abs(img1["img"] - img2["img"]),dim=(-1,-2,-3)) / torch.mean(torch.abs(ws1 - ws2),dim=(-1,-2,-3))
        # loss2 = torch.mean(torch.abs(img1["freq"][-1] - img2["freq"][-1]),dim=(-1,-2,-3)) / torch.mean(torch.abs(ws1 - ws2),dim=(-1,-2,-3))
        loss_lz = 1 / (loss1 + eps)
        return loss_lz

    def forward(self,x):
        ws = self.make_style(batch_size=x.size(0))
        gs, en_feats = self.encoder(x,ws)
        ws = ws.unsqueeze(1).repeat(1, self.latent_num, 1)
        gen_out = self.generator(ws, en_feats)
        return gen_out,ws

    def multiple_forward(self,x,sample_num = 3, mask_ratio = 0.8):
        self.encoder.mask_ratio = mask_ratio
        out, w1 = self(x)
        bt = max(2 * sample_num, 8)
        input_imgs_ = x.repeat(bt, 1, 1, 1)
        out_, w_ = self(input_imgs_)
        scores = self.get_scores(out, out_, w1, w_)
        topk_values, topk_indices = torch.topk(scores, k=sample_num - 1, largest=False)
        out_img = torch.cat([out['img'], torch.index_select(out_["img"], dim=0, index=topk_indices)], dim=0)
        return out_img

    @torch.no_grad()
    def infer(self,imgs,masks,sample_num = 3, mask_ratio = 0.8):
        input_imgs = self.preprocess(imgs,masks)
        out_imgs = self.multiple_forward(input_imgs,sample_num = sample_num, mask_ratio = mask_ratio)
        comp_imgs = self.imgs_t * self.masks_t + out_imgs * (1 - self.masks_t)
        comp_imgs = self.postprocess(comp_imgs)
        return comp_imgs

    @torch.no_grad()
    def gan_infer(self, batch_size):
        out = self.gan_forward(batch_size)
        out = self.dwt_to_img(out)
        out = self.postprocess(out)
        return out

    def img_to_dwt(self, img):
        low, high = self.dwt(img)
        b, _, _, h, w = high[0].size()
        high = high[0].view(b, -1, h, w)
        freq = torch.cat([low, high], dim=1)
        return freq

    def dwt_to_img(self, img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return self.idwt((low, [high]))

class MobileFill_v2(nn.Module):
    def __init__(self,device= 'cuda',target_size=512, input_nc=4,down_num = 5, latent_nc=512,mlp_layers=8):
        super(MobileFill_v2, self).__init__()
        self.channels = {
            4: 256,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 64,
        }
        en_channels = [v for k, v in self.channels.items() if k < target_size][::-1]
        gen_channels = [v for k,v in self.channels.items() if k >= (target_size // 2**down_num) and (k <= target_size //2)]
        self.device = device
        self.encoder = EESPNet_v21(config=en_channels, input_nc=input_nc, output_nc=latent_nc, input_size=target_size, down_num=down_num).to(device)
        # self.encoder = LFFC_encoder(input_nc=input_nc,latent_nc=latent_nc,ngf=64,n_downsampling=4,n_blocks=4).to(device)
        # self.encoder = mobilevit_xs(num_classes=latent_nc).to(device)
        self.mapping_net = MappingNetwork(style_dim=latent_nc,n_layers=mlp_layers).to(device)
        # self.generator = StyleEncoder_v6(channels= gen_channels, latent_nc = latent_nc, device=device).to(device)
        self.generator = MobileSynthesisNetwork_v5(style_dim=latent_nc,channels=gen_channels,device=device).to(device)
        # self.co_mod_layer = nn.Linear(in_features=2 * latent_nc, out_features= latent_nc).to(self.device)
        self.latent_nc = latent_nc
        self.latent_num = self.generator.wsize()
        self.latent_num_gan = 2 * int(math.log(target_size,2)) - 2
        self.dwt = DWTForward(J=1, mode='zero', wave='db1').to(self.device)
        self.idwt = DWTInverse(mode="zero", wave="db1").to(self.device)

    def preprocess(self,imgs,masks):
        if not isinstance(imgs,list):
            imgs = [imgs]
            masks = [masks]

        for i in range(len(imgs)):
            if isinstance(imgs[i],str):
                imgs[i] = np.array(Image.open(imgs[i]))
                masks[i] = np.array(Image.open(masks[i]))

            if len(masks[i].shape) < 3:
                masks[i] = np.expand_dims(masks[i],axis=-1)

        imgs_t = cv2tensor(imgs).to(self.device)
        masks_t = cv2tensor(masks).to(self.device)

        if masks_t.size(1) > 1:
            masks_t = masks_t[:, 2:3, :, :]

        self.masks_t = 1 - masks_t      #0 for holes
        self.imgs_t = imgs_t / 0.5 - 1.0

        # self.imgs_t = self.img_to_dwt(self.imgs_t)
        self.masked_imgs = self.imgs_t * self.masks_t
        input_imgs = torch.cat((self.masked_imgs, self.masks_t), dim=1)
        return input_imgs

    def postprocess(self,imgs_t,to_cv=True):
        imgs_t = (imgs_t + 1.0) * 0.5  #scale to 0 ~ 1
        if not to_cv:
            return imgs_t

        return tensor2cv(imgs_t)

    @torch.no_grad()
    def compute_mean_style(self, style_dim, wsize=1, batch_size=4096):
        style = self.mapping_net(torch.randn(batch_size, style_dim)).mean(0, keepdim=True)
        if wsize != 1:
            style = style.unsqueeze(1).repeat(1, wsize, 1)
        return style

    def mixing_style(self, batch_size, n_images = 4096):
        noise1 = torch.randn(batch_size, self.latent_nc).to(self.device)
        style1 = self.mapping_net(noise1)

        noise2 = torch.randn(batch_size, self.latent_nc).to(self.device)
        style2 = self.mapping_net(noise2)

        diff = (style2 - style1) / n_images

        return style1 + diff * random.randint(0, n_images)

    def make_style(self,batch_size):
        noise = torch.randn(batch_size, self.latent_nc).to(self.device)
        style = self.mapping_net(noise)
        return style

    def mode_seeking_loss(self, img1, img2, ws1, ws2):
        eps = 1 * 1e-5
        loss1 = torch.mean(torch.abs(img1["img"] - img2["img"])) / torch.mean(torch.abs(ws1 - ws2))
        loss2 = torch.mean(torch.abs(img1["freq"][-1] - img2["freq"][-1])) / torch.mean(torch.abs(ws1 - ws2))
        loss_lz = 1 / (loss1 + + loss2 + eps)
        return loss_lz

    def get_scores(self,img1,img2,ws1,ws2):
        eps = 1 * 1e-5
        loss1 = torch.mean(torch.abs(img1["img"] - img2["img"]),dim=(-1,-2,-3)) / torch.mean(torch.abs(ws1 - ws2),dim=(-1,-2,-3))
        # loss2 = torch.mean(torch.abs(img1["freq"][-1] - img2["freq"][-1]),dim=(-1,-2,-3)) / torch.mean(torch.abs(ws1 - ws2),dim=(-1,-2,-3))
        loss_lz = 1 / (loss1 + eps)
        return loss_lz

    def forward(self,x):
        ws = self.make_style(batch_size=x.size(0))
        cs, en_feats = self.encoder(x, ws)
        ws = ws.unsqueeze(1).repeat(1, self.latent_num, 1)
        cs = cs.unsqueeze(1).repeat(1, self.latent_num_gan, 1)
        gen_out = self.generator(ws, en_feats)
        return gen_out,cs

    def multiple_forward(self,x,sample_num = 3, mask_ratio = 0.8):
        self.encoder.mask_ratio = mask_ratio
        out, w1 = self(x)
        bt = max(2 * sample_num, 8)
        input_imgs_ = x.repeat(bt, 1, 1, 1)
        out_, w_ = self(input_imgs_)
        scores = self.get_scores(out, out_, w1, w_)
        topk_values, topk_indices = torch.topk(scores, k=sample_num - 1, largest=False)
        out_img = torch.cat([out['img'], torch.index_select(out_["img"], dim=0, index=topk_indices)], dim=0)
        return out_img

    @torch.no_grad()
    def infer(self,imgs,masks,sample_num = 3, mask_ratio = 0.8):
        input_imgs = self.preprocess(imgs,masks)
        # out_imgs = self.multiple_forward(input_imgs,sample_num = sample_num, mask_ratio = mask_ratio)
        out_imgs,cs = self.forward(input_imgs)
        comp_imgs = self.imgs_t * self.masks_t + out_imgs['img'] * (1 - self.masks_t)
        # comp_imgs = self.postprocess(comp_imgs)
        return comp_imgs,cs

    @torch.no_grad()
    def gan_infer(self, batch_size):
        out = self.gan_forward(batch_size)
        out = self.dwt_to_img(out)
        out = self.postprocess(out)
        return out

    def img_to_dwt(self, img):
        low, high = self.dwt(img)
        b, _, _, h, w = high[0].size()
        high = high[0].view(b, -1, h, w)
        freq = torch.cat([low, high], dim=1)
        return freq

    def dwt_to_img(self, img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return self.idwt((low, [high]))

if __name__ == '__main__':
    from complexity import *
    device = "cuda"
    m = torch.randn(1,1,256,256).to(device)
    x = torch.randn(1,4,256,256).to(device)
    ws = torch.randn(1,512).to(device)
    en_x = torch.randn(1,512,16,16).to(device)
    mask = torch.randn(1,1,256,256).to(device)

    model = MobileFill(input_nc=4,target_size=x.size(-1))
    # model = MobileFill_v2(input_nc=4, target_size=x.size(-1))
    # style_square, style, _ = model.style_encoder(ws)
    styles, en_feats = model.encoder(x,ws)
    out,cs = model(x)
    print_network_params(model,"MobileFill")
    print_network_params(model.encoder,"MobileFill.encoder")
    print_network_params(model.generator,"MobileFill.generator")
    print_network_params(model.mapping_net, "MobileFill.mapping_net")

    flops = 0.0
    flops += flop_counter(model.encoder,(x,ws))
    flops += flop_counter(model.mapping_net, ws)
    styles = styles.unsqueeze(1).repeat(1, 19, 1)
    flops += flop_counter(model.generator,(styles, en_feats))
    print(f"Total FLOPs: {flops:.5f} G")


