from src.models.encoder import StyleEncoder as Encoder
from src.models.encoder import RefineEncoder
from src.models.decoder import Decoder
from src.models.generator import MobileSynthesisNetwork_v7
import torch.nn as nn
from PIL import Image
from src.utils.util import tensor2cv,cv2tensor
import torch
import random
import numpy as np


class MobileFill_v3(nn.Module):
    def __init__(self,device= 'cuda',target_size=512, input_nc=4,down_num = 4, latent_nc=512, sample_radius = 0.001):
        super(MobileFill_v3, self).__init__()
        en_channels = [128, 256, 256, 512]
        refine_en_channels = [64, 64, 128, 256, 512]
        gen_channels = [512, 512, 256, 256, 128, 64]
        self.device = device
        self.out_im_size = target_size // 2 ** down_num
        self.encoder = Encoder(config = en_channels,input_nc = input_nc,input_size = target_size,down_num=down_num).to(device)
        self.generator = MobileSynthesisNetwork_v7(style_dim=latent_nc,channels=gen_channels, device=device).to(device)
        self.refine_encoder = RefineEncoder(config=refine_en_channels,input_nc=3,input_size=target_size,down_num=down_num).to(device)
        self.refine_decoder = Decoder(in_dim=en_channels[-1],out_dim=3,up_blocks=down_num).to(device)
        self.radius = nn.Parameter(torch.tensor(sample_radius,device=self.device),requires_grad=self.training)
        self.latent_nc = latent_nc
        self.latent_num = self.encoder.n_latent

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

        masks_t = 1 - masks_t      #0 for holes
        imgs_t = imgs_t / 0.5 - 1.0

        # self.imgs_t = self.img_to_dwt(self.imgs_t)
        # masked_imgs = imgs_t * masks_t
        # input_imgs = torch.cat((masked_imgs, masks_t), dim=1)
        return imgs_t, masks_t

    def postprocess(self,imgs_t,to_cv=True):
        imgs_t = (imgs_t + 1.0) * 0.5  #scale to 0 ~ 1
        if not to_cv:
            return imgs_t

        return tensor2cv(imgs_t)

    def get_radius(self,query):
        min_val = -self.radius
        max_val = self.radius
        random_tensor = torch.rand(query.shape, device = self.device)
        scaled_tensor = 2 * (random_tensor - 0.5)  # scale to [-1, 1]
        radius = (scaled_tensor * max_val).clamp(min_val, max_val)
        return radius

    def get_latents_pool(self, input_x):
        query = self.make_style(input_x)
        # bt, n, dim = query.shape
        # num_neg = int(neg_ratio * 16)
        latents = [query]
        poses = []
        latent_num = 16 if self.out_im_size < 32 else 64
        for i in range(latent_num -1):
            pos = self.get_radius(query)
            poses.append(pos.add_(query))

        latents += poses

        # if num_neg > 0:
        #     negs = []
        #     for k in range(num_neg):
        #         neg = self.make_style(input_x)
        #         while (neg - query).abs().min() < radius:
        #             neg = self.make_style(input_x)
        #         negs.append(neg)
        #
        #     latents += negs

        return latents

    def latent_sampling(self, input_x):
        latents = self.get_latents_pool(input_x = input_x)
        if np.random.binomial(1,0.5) > 0:
            idx = random.randint(1,len(latents) - 1)
            selected_latent = latents[idx]
        else:
            idx = 0

        hidden = torch.cat(latents, dim = 1).permute(0,2,1).view(input_x.size(0),-1, self.out_im_size , self.out_im_size)
        return latents, hidden

    def make_style(self,input_x):
        noise = torch.randn((input_x.size(0), self.latent_nc), device = self.device)
        style,*_ = self.encoder(input_x,noise)
        if self.latent_num == 16:
            return style
        else:
            return style[:,2:]

    def forward(self, x, mask, noise = None):
        if noise == None:
            noise = torch.randn((x.size(0), 512), device=self.device)

        if mask.size(1) == 3:
            mask = mask[:,2:3]
        input_x = torch.cat([x * mask, mask], dim = 1)
        latents, detals, en_feats = self.encoder(input_x, noise)
        coarse_x = self.generator(latents)
        merged_x = x * mask + coarse_x * (1 - mask)
        en_x,feats = self.refine_encoder(merged_x)
        out_x = self.refine_decoder(en_x,feats)
        return out_x, detals, en_feats

    def get_latent(self, x, mask, noise = None):
        if noise == None:
            noise = torch.randn((x.size(0), 512), device=self.device)

        if mask.size(1) == 3:
            mask = mask[:, 2:3]
        input_x = torch.cat([x * mask, mask], dim=1)
        latents, deltas, en_feats = self.encoder(input_x, noise)
        return latents, deltas, en_feats

    def gan_forward(self,latents, en_feats):
        x = self.generator(latents, en_feats)
        return x

    def refine(self, merged_x):
        en_x, feats = self.refine_encoder(merged_x)
        out_x = self.refine_decoder(en_x, feats)
        return out_x

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


if __name__ == '__main__':
    from complexity import *
    device = "cpu"
    x = torch.randn(1,3,256,256).to(device)
    target =  torch.randn(1, 3, 256, 256).to(device)
    input_x = torch.randn(1, 4, 256, 256).to(device)
    ws = torch.randn(1,512).to(device)
    en_x = torch.randn(1,512,16,16).to(device)
    mask = torch.randn(1,1,256,256).to(device)
    en_feats1 = [
        torch.randn(1, 512, 16, 16).to(device),
        torch.randn(1, 256, 32, 32).to(device),
        torch.randn(1, 256, 64, 64).to(device),
        torch.randn(1, 128, 128, 128).to(device)
    ]

    en_feats2 = [
        torch.randn(1, 512, 16, 16).to(device),
        torch.randn(1, 256, 32, 32).to(device),
        torch.randn(1, 128, 64, 64).to(device),
        torch.randn(1, 64, 128, 128).to(device)
    ]

    # model = MobileFill(input_nc=4,target_size=x.size(-1))
    # # model = MobileFill_v2(input_nc=4, target_size=x.size(-1))
    # # style_square, style, _ = model.style_encoder(ws)
    # styles, en_feats = model.encoder(x,ws)
    # out,cs = model(x)
    # print_network_params(model,"MobileFill")
    # print_network_params(model.encoder,"MobileFill.encoder")
    # print_network_params(model.generator,"MobileFill.generator")
    # print_network_params(model.mapping_net, "MobileFill.mapping_net")
    #
    # flops = 0.0
    # flops += flop_counter(model.encoder,(x,ws))
    # flops += flop_counter(model.mapping_net, ws)
    # styles = styles.unsqueeze(1).repeat(1, 19, 1)
    # flops += flop_counter(model.generator,(styles, en_feats))
    # print(f"Total FLOPs: {flops:.5f} G")

    model = MobileFill_v3(input_nc=4, target_size=x.size(-1), device=device)
    out_x = model(x, mask)
    latents, *_ = model.get_latent(x, mask, ws)
    print_network_params(model, "MobileFill_v3")
    print_network_params(model.encoder, "MobileFill.encoder")
    print_network_params(model.generator, "MobileFill.generator")
    print_network_params(model.refine_encoder, "MobileFill.refine_encoder")
    print_network_params(model.refine_decoder, "MobileFill.refine_decoder")

    flops = 0.0
    flops += flop_counter(model.encoder, (input_x, ws))
    flops += flop_counter(model.generator, (latents, ))
    flops += flop_counter(model.refine_encoder, (x, ))
    flops += flop_counter(model.refine_decoder, (en_x, en_feats2))
    print(f"Total FLOPs: {flops:.5f} G")


    # model.train()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # import torch.nn.functional as F
    # test_data = [torch.randn(1,3,256,256), torch.randn(1,3,256,256), torch.randn(1,3,256,256)]
    # for i, d in enumerate(test_data):
    #     d = d.to(device)
    #     optimizer.zero_grad()
    #     out_x = model(d, mask)
    #     loss = F.mse_loss(out_x, target)
    #     loss.backward()
    #     # 检查 radius 是否参与反向传播的计算
    #     print(f"epoch {i}, radius grad: {model.radius.grad}")
    #     optimizer.step()


