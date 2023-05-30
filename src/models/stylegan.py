from src.stylegan2.model import Generator as styleGAN
import torch
from PIL import Image
from src.utils.util import tensor2cv

class StyleGANModel():
    def __init__(self,model_path,targetSize=256, device = 'cuda'):
        self.device = device
        self.model = styleGAN(size=targetSize,style_dim=512,n_mlp=8)
        self.model.load_state_dict(torch.load(model_path)["g_ema"])
        self.model.eval().requires_grad_(False)
        self.model = self.model.to(device)

    def sample(self,sample_z):
        sample_z = sample_z.to(self.device)
        samples,multi_scale_imgs,latents = self.model(
            [sample_z],return_latents=True, truncation=1.0, truncation_latent=None
        )
        return samples,latents

    def sample_with_latent(self, z):
        samples = self.model.simple_forward(z)
        return samples

    def postprocess(self,imgs):
        imgs = (imgs + 1.0) / 2.0
        return imgs


if __name__ == '__main__':
    model_path = "/home/codeoops/CV/MobileFill/src/stylegan2/checkpoint/210000.pt"
    model = StyleGANModel(model_path = model_path)
    multi_scale_imgs,latents = model.sample(batchSize=3)
    for imgs in multi_scale_imgs:
        imgs_list = tensor2cv(model.postprocess(imgs))
        Image.fromarray(imgs_list[0]).show()
    print()


