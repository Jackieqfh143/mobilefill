from src.MAT.networks.mat import Generator
import src.MAT.dnnlib as dnnlib
import src.MAT.legacy as legacy
import numpy as np
import torch

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


class MAT():

    def __init__(self,model_path,device,targetSize=256,**kwargs):
        super(MAT, self).__init__()
        self.device = torch.device(device)
        # comp_opt = kwargs['comp_opt']
        with dnnlib.util.open_url(model_path) as f:
            G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False)  # type: ignore
        self.G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=targetSize, img_channels=3).eval().requires_grad_(False)
        copy_params_and_buffers(G_saved, self.G, require_all=True)
        self.G = self.G.to(self.device)

        # no Labels.
        self.label = torch.zeros([1, self.G.c_dim], device=self.device)
        print('MAT loaded.')

    @torch.no_grad()
    def forward(self,imgs,masks):
       # if imgs.shape[3] != 512:
       #     noise_mode = 'random'
      #  else:
       #     noise_mode = 'const'

        noise_mode = 'random'
        z = torch.randn(imgs.size(0), self.G.z_dim).to(self.device)
        output = self.G(imgs, masks, z, self.label, truncation_psi=1, noise_mode=noise_mode)

        return output

    @torch.no_grad()
    def infer_with_np(self,imgs_np,masks_np,**kwargs):
        comp_imgs = []
        for im,mask in zip(imgs_np,masks_np):
            image_t,mask_t = self.preprocess(im,mask)
            output_t,z = self.forward(image_t,mask_t)
            output_np = self.postprocess(output_t)
            comp_imgs.append(output_np)

        return comp_imgs

    def preprocess(self,im,mask):
        im = im.transpose(2, 0, 1)
        if len(mask.shape) < 3:
            mask = np.expand_dims(mask, axis=-1)
        mask = mask.transpose(2, 0, 1)
        image = (torch.from_numpy(im).float().to(self.device) / 127.5 - 1).unsqueeze(0)
        mask = mask.astype(np.float32) / 255.0
        mask = torch.from_numpy(mask).float().to(self.device).unsqueeze(0)
        mask = 1 - mask   #0 for holes
        return image,mask

    def postprocess(self,img):
        img = (img.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
        return img[0].cpu().numpy()



if __name__ == '__main__':
    model_path = "/home/codeoops/CV/MobileFill/checkpoints/MAT/celeba-hq/CelebA-HQ_512.pkl"
    img = torch.randn((1,3,512,512),device="cuda")
    mask = torch.randn((1,1,512,512),device="cuda")
    model = MAT(model_path=model_path,device="cuda",targetSize=512)
    out = model.forward(img,mask)




