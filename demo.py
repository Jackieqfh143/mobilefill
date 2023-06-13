import torch
from src.models.mobileFill import MobileFill
import torch.nn.functional as F
import numpy as np
from PIL import Image
import time
from src.utils.util import tensor2cv
if __name__ == '__main__':
    model_path = './checkpoints/G-step=268000_lr=0.0001_default_loss=0.4916.pth'
    test_img_path = './example/imgs/00001_im_truth.jpg'
    test_mask_path = './example/masks/00001_im_mask.png'

    test_img = np.array(Image.open(test_img_path))
    test_mask = np.array(Image.open(test_mask_path))
    device = 'cpu'
    model = MobileFill(device=device,input_nc=4)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
   # style = torch.randn(1,512)
 #   out_dict = model.infer([test_img],[test_mask],styles=None,return_dict=True)

  #  en_x = out_dict['en_x']
   # style_ = out_dict['style']
   # co_style = out_dict['co_style']
   # out_img = out_dict['results']

   # co_style_en_x_loss = F.mse_loss(en_x,co_style)
   # co_style_style_loss = F.mse_loss(co_style,style_)


   # print(f"co_style_en_x_loss: {co_style_en_x_loss}")
   # print(f"co_style_style_loss: {co_style_style_loss}")
    # Image.fromarray(out_img[0]).show()


   # styles = torch.randn(3,512)
    for i in range(3):
       # style = styles[i:i+1]
        start_time = time.time()
        out_np = model.infer([test_img],[test_mask],styles=None,return_dict=False)

        Image.fromarray(out_np[0]).show()
        time_span = time.time() - start_time

        print(f"{device} inference time cost: {time_span * 1000} ms")
