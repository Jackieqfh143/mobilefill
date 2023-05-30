import torch
from src.models.mobileFill import MobileFill
from src.evaluate.evaluation import validate
from tqdm import tqdm
import numpy as np
from PIL import Image
import time
import glob
import os

def make_iter(data,batch_size):
    idx = 0
    out = []
    last_bt = len(data) % batch_size
    break_flag = int(len(data) / batch_size) * batch_size
    assert last_bt + break_flag == len(data)
    while True:
        out.append(data[idx:idx+batch_size])
        idx += batch_size
        if idx >= break_flag:
            break

    if last_bt > 0:
        out.append(data[break_flag:])

    return iter(out),len(data)

def load_model(device='cuda'):
    model = MobileFill(device=device, input_nc=4)
    model.eval().requires_grad_(False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device)

    return model

def data_prepare(img_path,mask_path,batch_size):
    imgs_path = glob.glob(img_path+"/*.png") + glob.glob(img_path+"/*.jpg")
    masks_path = glob.glob(mask_path+"/*.png") + glob.glob(mask_path+"/*.jpg")
    imgs_path.sort()
    masks_path.sort()
    imgs = []
    masks = []

    for im_path,m_path in zip(imgs_path,masks_path):
        im = np.array(Image.open(im_path))
        m = np.array(Image.open(m_path))
        imgs.append(im)
        masks.append(m)

    img_iter,data_len = make_iter(imgs,batch_size)
    mask_iter,_ = make_iter(masks,batch_size)

    return img_iter,mask_iter,data_len

def save_results(imgs,save_path,idx):
    for i in range(len(imgs)):
        save_name = save_path + f"/{idx+i}.png"
        Image.fromarray(imgs[i]).save(save_name)


if __name__ == '__main__':
    device = "cuda"
    data_name = "celeba-hq"
    model_path = './checkpoints/G-step=202000_lr=0.0001_default_loss=0.4908.pth'
    img_path = '/home/codeoops/CV/InPainting/Inpainting_baseline/compare/results/celeba-hq-256/(thick_256_5.0k)/real_imgs'
    mask_path = '/home/codeoops/CV/InPainting/Inpainting_baseline/compare/results/celeba-hq-256/(thick_256_5.0k)/masks'
    save_dir = f'./results/{data_name}/fake_imgs'
    save_paths = []
    n_samples = 3
    batch_size = 10

    for i in range(n_samples):
        path = save_dir+f"_{i}"
        if not os.path.exists(path):
            os.makedirs(path)
        save_paths.append(path)
    #
    #
    # img_iter,mask_iter,data_len = data_prepare(img_path,mask_path,batch_size)
    # model = load_model(device)
    # idx = 0
    # time_span = 0.0
    # start_time = time.time()
    # for imgs,masks in tqdm(zip(img_iter,mask_iter)):
    #     for i in range(n_samples):
    #         out_imgs = model.infer(imgs, masks, return_dict=False)
    #         save_results(out_imgs,save_path=save_paths[i],idx=idx)
    #     idx += len(imgs)
    # time_span += time.time() - start_time
    # time_span = time_span / (data_len * n_samples) * 1e3
    #
    # print(f"Average infer time span for each img on {device}: {time_span} ms")

    for i in range(n_samples):
        validate(img_path,save_paths[i],real_suffix="_truth",fake_suffix="")