import torch
from src.models.mobileFill import MobileFill,MobileFill_v2
import numpy as np
from PIL import Image
import time
from src.utils.util import tensor2cv
from src.models.stylegan import StyleGANModel
from torchvision.utils import make_grid
if __name__ == '__main__':
    model_path = '/home/codeoops/CV/MobileFill/checkpoints/G-step=44000_lr=0.00057_ema_loss=0.5046.pth'
    stylegan_path = "/home/codeoops/CV/MobileFill/src/stylegan2/checkpoint/210000.pt"
    test_img_path = './example/imgs/00044_im_truth.jpg'
    test_mask_path = './example/masks/00044_im_mask.png'

    test_img = np.array(Image.open(test_img_path))
    test_mask = np.array(Image.open(test_mask_path))
    device = 'cuda'
    styleGAN = StyleGANModel(model_path = stylegan_path, device= device)
    model = MobileFill_v2(device=device,input_nc=4,target_size=256)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    img_list = []
    sample_num = 3
    time_span = 0
    for i in range(sample_num):
        start_time = time.time()
        out,cs = model.infer([test_img],[test_mask])
        out = (out + 1.0) * 0.5
        # gan_out = model.postprocess(styleGAN.sample_with_latent(cs))
        # Image.fromarray(gan_out[0]).show()
        # Image.fromarray(out_np[0]).show()
        img_list.append(out)
        time_span = time.time() - start_time

    img_gird = make_grid(torch.cat(img_list, dim = 0))
    Image.fromarray(tensor2cv(img_gird.unsqueeze(0))[0]).show()
    print(f"{device} Average inference time cost: {time_span * 1000 / sample_num} ms")
