import cv2
import src.FcF.dnnlib as dnnlib
import torch
import src.FcF.legacy as legacy
from PIL import Image
import numpy as np


class FCFModel():

    def __init__(self,model_path, device='cpu', targetSize=256):
        super(FCFModel, self).__init__()
        self.device = torch.device(device)
        self.target_size = targetSize
        with dnnlib.util.open_url(model_path) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False)  # type: ignore

        # Labels.
        self.label = torch.zeros([1, self.G.c_dim], device=self.device)
        class_idx = None
        if self.G.c_dim != 0:
            if class_idx is None:
                print('Must specify class label with --class when using a conditional network')
            self.label[:, class_idx] = 1
        else:
            if class_idx is not None:
                print('warn: --class=lbl ignored when running on an unconditional network')

        print('FCF loaded.')


    def preprocess(self,imgs_path,masks_path):
        out_imgs = []
        out_masked_imgs = []
        out_masks =[]
        for im_path,mask_path in zip(imgs_path,masks_path):
            img_array = np.array(Image.open(im_path).convert('RGB'))
            mask_array = np.array(Image.open(mask_path).convert('L')) / 255

            mask_array = cv2.resize(mask_array,(self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
            img_array = cv2.resize(img_array, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

            img = img_array.transpose((2, 0, 1))
            mask_tensor = torch.from_numpy(mask_array).to(torch.float32)
            mask_tensor = mask_tensor.unsqueeze(0).to(self.device)

            img = torch.from_numpy(img.astype(np.float32))
            img = (img.to(torch.float32) / 127.5 - 1).to(self.device)


            img_masked = img.clone()
            img_masked = img_masked * (1 - mask_tensor)
            img_masked = img_masked.to(torch.float32)

            out_imgs.append(img.unsqueeze(0))
            out_masked_imgs.append(img_masked.unsqueeze(0))
            out_masks.append(mask_tensor.unsqueeze(0))

        out_imgs = torch.cat(out_imgs,dim=0)
        out_masked_imgs = torch.cat(out_masked_imgs,dim=0)
        out_masks = torch.cat(out_masks,dim=0)

        return out_imgs,out_masked_imgs,out_masks

    def postprocess(self,imgs_t):
        out_imgs = []
        for i in range(imgs_t.shape[0]):
            img = imgs_t[i]
            lo, hi = [-1, 1]
            img = np.asarray(img.cpu(), dtype=np.float32).transpose(1, 2, 0)
            img = (img - lo) * (255 / (hi - lo))
            img = np.rint(img).clip(0, 255).astype(np.uint8)
            out_imgs.append(img)

        return out_imgs

    def infer_with_np(self, imgs_path, masks_path):
        img, img_masked, mask_tensor = self.preprocess(imgs_path, masks_path)
        comp_imgs = self.forward(img, 1 - mask_tensor)
        comp_imgs = self.postprocess(comp_imgs)

        return comp_imgs


    @torch.no_grad()
    def forward(self,real_imgs,masks):
        masks = 1 - masks    #1 for holes
        img_masked = real_imgs.clone()
        img_masked = img_masked * (1 - masks)
        img_masked = img_masked.to(torch.float32)
        pred_im = self.G(img=torch.cat([0.5 - masks, img_masked], dim=1), c=self.label, truncation_psi=0.1,
                     noise_mode='const')
        comp_imgs = masks * pred_im + (1 - masks) * real_imgs

        return comp_imgs



if __name__ == '__main__':
    import torch
    import time
    FCF_model_path = "/home/codeoops/CV/MobileFill/checkpoints/FCF/celeba-hq/celeba-hq.pkl"
    test_img_path = '/home/codeoops/CV/MobileFill/example/imgs/00044_im_truth.jpg'
    test_mask_path = '/home/codeoops/CV/MobileFill/example/masks/00044_im_mask.png'
    device = 'cuda'
    FCF_model = FCFModel(model_path=FCF_model_path,device=device)
    n_samples = 3  # how many different outputs for each input images
    time_span = 0.0
    for i in range(n_samples):
        start_time = time.time()
        out = FCF_model.infer_with_np([test_img_path],[test_mask_path])
        time_span += time.time() - start_time
        Image.fromarray(out[0]).show()

    time_span = time_span / n_samples * 1000
    print(f"Average infer time span per img on {device}: {time_span:.2f} ms")