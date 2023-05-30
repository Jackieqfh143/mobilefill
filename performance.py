from torch.utils.data import DataLoader
from src.evaluate.evaluation import validate
from prefetch_generator import BackgroundGenerator
from src.models.mobileFill import MobileFill
from collections import OrderedDict
from tqdm import tqdm
import random
import warnings
import argparse
import torch
import time
import os
import traceback
from torch.utils import data
from src.utils.im_process import *
from src.utils.visualize import show_im2_html
import torchvision.transforms.functional as F
import glob

parse = argparse.ArgumentParser()
parse.add_argument('--device',type=str,dest='device',default="cuda",help='device')
parse.add_argument('--dataset_name',type=str,dest='dataset_name',default="Place",help='dataset name')
parse.add_argument('--model_path',type=str,dest='model_path',default="./checkpoints/place_best.pth",help='model path')
parse.add_argument('--mask_type',type=str,dest='mask_type',default="thick_256",help='the mask type')
parse.add_argument('--batch_size',type=int,dest='batch_size',default=20,help='batch size')
parse.add_argument('--target_size',type=int,dest='target_size',default=256,help='target image size')
parse.add_argument('--random_seed',type=int,dest='random_seed',default=2023,help='random seed')
parse.add_argument('--total_num',type=int,dest='total_num',default=10000,help='total number of test images')
parse.add_argument('--sample_num',type=int,dest='sample_num',default=3,help='how many different output images for one input')
parse.add_argument('--rand_ratio',type=int,dest='rand_ratio',default=0.8,help='Higher ratio means more higher image diversity')
parse.add_argument('--img_dir',type=str,dest='img_dir',default="",help='sample images for validation')
parse.add_argument('--mask_dir',type=str,dest='mask_dir',default="",help='sample masks for validation')
parse.add_argument('--save_dir',type=str,dest='save_dir',default="./results",help='path for saving the results')
parse.add_argument('--aspect_ratio_kept', action='store_true',help='keep the image aspect ratio when resize it')
parse.add_argument('--fixed_size', action='store_true',help='fixed the crop size')
parse.add_argument('--center_crop', action='store_true',help='center crop')
arg = parse.parse_args()


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class ValDataSet(data.Dataset):
    def __init__(self,img_dir,mask_dir,total_num):
        super(ValDataSet, self).__init__()
        self.imgs = sorted(glob.glob(img_dir + "/*.jpg") + glob.glob(img_dir + "/*.png"))
        self.masks = sorted(glob.glob(mask_dir + "/*.jpg") + glob.glob(mask_dir + "/*.png"))


        max_num = min(len(self.imgs),total_num)
        self.imgs = self.imgs[:max_num]
        self.masks = self.masks[:max_num]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            traceback.print_exc()
            print('loading error: ' + self.imgs[index])
            item = self.load_item(0)

        return item

    def load_item(self,idx):
        input = self.preprocess(self.imgs[idx],self.masks[idx])
        return input

    #propocess one image each time
    def preprocess(self,img_path,mask_path):
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        # img = resize(img)
        # mask = resize(mask)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
            mask = np.concatenate((mask, mask, mask), axis=-1)

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
            img = np.concatenate((img, img, img), axis=-1)

        mask = mask.astype(np.uint8)

        mask = (mask > 0).astype(np.uint8) * 255
        img_t_raw = F.to_tensor(img).float()
        mask_t = F.to_tensor(mask).float()

        mask_t = mask_t[2:3,:,:]
        mask_t = 1 - mask_t    #set holes = 0
        img_t = img_t_raw / 0.5 - 1.0
        masked_im = img_t * mask_t
        input_x = torch.cat((masked_im,mask_t),dim=0)

        return input_x,img_t_raw,mask_t

def post_process(out,gt,mask,idx,save_path,sample_num):
    masked_im = gt * mask
    for i in range(gt.size(0)):
        gt_img_np = gt[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        mask_np = (1 - mask[i]).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        masked_img_np, _ = get_transparent_mask(gt_img_np, mask_np)  #the input mask should be 1 for holes
        Image.fromarray(masked_img_np).save(save_path + f'/{i + idx :0>5d}_im_masked.jpg')
        Image.fromarray(gt_img_np).save(save_path + f'/{i + idx:0>5d}_im_truth.jpg')


        for j in range(sample_num):
            out[i][j] = (out[i][j] + 1.0) * 0.5
            comp_im = out[i][j] * (1 - mask[i]) + masked_im[i]
            fake_img_np = comp_im.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            Image.fromarray(fake_img_np).save(save_path + f'/{i + idx:0>5d}_im_out_{j}.jpg')


def resize(img):
    if arg.aspect_ratio_kept:
        imgh, imgw = img.shape[0:2]
        side = np.minimum(imgh, imgw)
        if arg.fixed_size:
            if arg.center_crop:
                # center crop
                j = (imgh - side) // 2
                i = (imgw - side) // 2
                img = img[j:j + side, i:i + side, ...]
            else:
                #random crop
                j = (imgh - side)
                i = (imgw - side)
                h_start = 0
                w_start = 0
                if j != 0:
                    h_start = random.randrange(0, j)
                if i != 0:
                    w_start = random.randrange(0, i)
                img = img[h_start:h_start + side, w_start:w_start + side, ...]
        else:
            if side <= arg.target_size:
                j = (imgh - side)
                i = (imgw - side)
                h_start = 0
                w_start = 0
                if j != 0:
                    h_start = random.randrange(0, j)
                if i != 0:
                    w_start = random.randrange(0, i)
                img = img[h_start:h_start + side, w_start:w_start + side, ...]
            else:
                side = random.randrange(arg.target_size, side)
                j = (imgh - side)
                i = (imgw - side)
                h_start = random.randrange(0, j)
                w_start = random.randrange(0, i)
                img = img[h_start:h_start + side, w_start:w_start + side, ...]
    img = np.array(Image.fromarray(img).resize(size=(arg.target_size, arg.target_size)))
    return img

def set_random_seed(random_seed=666,deterministic=False):
    if random_seed is not None:
        print("Set random seed as {}".format(random_seed))
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        if deterministic:
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            #for faster training
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

def load_model():
    model = MobileFill(input_nc=4,target_size=arg.target_size)
    net_state_dict = model.state_dict()
    state_dict = torch.load(arg.model_path, map_location='cpu')
    new_state_dict = {k: v for k, v in state_dict.items() if k in net_state_dict}
    model.load_state_dict(OrderedDict(new_state_dict), strict=False)
    model.eval().requires_grad_(False).to(arg.device)

    return model

if __name__ == '__main__':
    inpaintingModel = load_model()
    save_dir = os.path.join(arg.save_dir,arg.dataset_name)
    save_path = os.path.join(save_dir,arg.mask_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # set_random_seed(arg.random_seed)

    test_dataset = ValDataSet(arg.img_dir,arg.mask_dir,arg.total_num)

    test_dataloader = DataLoaderX(test_dataset,
                                 batch_size=arg.batch_size, shuffle=False, drop_last=False,
                                 num_workers=8,
                                 pin_memory=True)


    time_span = 0.0

    print("Processing images...")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader)):
            input_x,gt,mask = batch
            input_x = input_x.to(arg.device)
            gt = gt.to(arg.device)
            mask = mask.to(arg.device)
            start_time = time.time()
            # out_imgs = []
            # for j in range(arg.sample_num):
            #     out = inpaintingModel(input_x)[0]["img"]
            #     out_imgs.append(out)
            #
            # out_imgs = torch.cat(out_imgs,dim=0)
            out_imgs_list = []
            for j in range(gt.size(0)):
                out_imgs = inpaintingModel.multiple_forward(input_x[j:j+1],sample_num=arg.sample_num,mask_ratio=arg.rand_ratio)
                out_imgs_list.append(out_imgs)
            time_span += (time.time() - start_time) / (input_x.size(0) * arg.sample_num)
            post_process(out_imgs_list,gt,mask,i * arg.batch_size,save_path,arg.sample_num)

    infer_speed = (time_span * 1000) / (arg.total_num)

    show_im2_html(web_title = f"Result_{arg.dataset_name}",
                  web_header = f"Inpainting Results on {arg.dataset_name}",
                  web_dir = save_dir,
                  img_dir = save_path,
                  im_size = arg.target_size,
                  max_num = 200)

    print(f"Inference speed: {infer_speed} ms/ img")

    print("Start Validating...")
    validate(real_imgs_dir=save_path,
            comp_imgs_dir=save_path,
            device=arg.device,
            get_FID=True,
            get_LPIPS=True,
            get_IDS=True)

