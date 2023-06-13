import os
import shutil
import torch
# import torch.nn as nn
# import numpy as np
from src.models.baseModel import BaseModel
from src.evaluate.loss import ResNetPL,l1_loss,Dis_loss_mask,Gen_loss,Dis_loss,VGG16FeatureExtractor
# Gen_loss_mask,smooth_l1_loss,l2_loss,\
#     ,featureMatchLoss,d_logistic_loss,\
    # g_nonsaturating_loss,d_r1_loss,l2_feat_mat
from src.utils.util import checkDir,tensor2cv
# from src.utils.diffaug import DiffAugment
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from PIL import Image
from torch_ema import ExponentialMovingAverage
from pytorch_wavelets import DWTInverse, DWTForward
from src.models.mobileFill import MobileFill

from src.models.encoder import StyleEncoder
from src.models.discriminator import MultidilatedNLayerDiscriminatorWithAtt, LatentCodesDiscriminator\
    # , MultidilatedNLayerDiscriminatorWithAtt_v2,\
    # MultidilatedNLayerDiscriminatorWithAtt_UNet,UNetDiscriminator,StyleGAN_Discriminator,EESPDiscriminator
from src.models.mat import MAT
# from src.models.fcf import FCFModel
# from src.stylegan2.model import Generator as styleGAN
# from src.models.stylegan import StyleGANModel
from collections import OrderedDict
from src.models.stylegan import StyleGANModel
import kornia

class InpaintingModel(BaseModel):
    def __init__(self,opt):
        super(InpaintingModel, self).__init__(opt)
        self.count = 0
        self.opt = opt
        self.mode = opt.mode
        self.lossNet = ResNetPL(weights_path=opt.lossNetDir) #segmentation network for calculate percptual loss
        self.flops = None
        self.device = self.accelerator.device
        self.lossNet = self.lossNet.to(self.device)
        self.recorder = SummaryWriter(self.log_path)
        self.current_lr = opt.lr
        self.current_d_lr = opt.d_lr

        if self.mode == 1:
            self.G_Net = MobileFill(input_nc=4,device=self.device,target_size=opt.targetSize)
            # self.G_Net = MobileFill_V2(input_nc=4,device=self.device,target_size=opt.targetSize)
            self.D_Net = MultidilatedNLayerDiscriminatorWithAtt(input_nc=3)
            self.G_opt = torch.optim.AdamW(self.G_Net.parameters(), opt.lr,
                                           betas=(opt.beta_g_min, opt.beta_g_max))
            self.D_opt = torch.optim.AdamW(self.D_Net.parameters(), lr=opt.d_lr,
                                           betas=(opt.beta_d_min, opt.beta_d_max))

            # self.load_from(opt.model_path)
            # self.G_Net.generator.eval().requires_grad_(False)
            # self.G_Net.mapping_net.eval().requires_grad_(False)
            #
            if opt.restore_training:
                self.load()

            if self.opt.enable_ema:
                self.ema_G = ExponentialMovingAverage(self.G_Net.parameters(), decay=0.995)
                self.ema_G.to(self.device)
                self.acc_args = [self.G_Net, self.D_Net, self.G_opt, self.D_opt]
            else:
                #args that should be prepared for accelerator
                self.acc_args = [self.G_Net, self.D_Net,self.G_opt, self.D_opt]


            if opt.enable_teacher:
                # self.teacher = styleGAN(size=opt.targetSize,style_dim=512,n_mlp=8).to(self.device)
                self.teacher = MAT(model_path=opt.teacher_path,device=self.device,targetSize=self.opt.targetSize)
                #
                # self.teacher = FCFModel(model_path=opt.teacher_path,device = self.device,targetSize=self.opt.targetSize)
                # self.img_augment = nn.Sequential(kornia.augmentation.RandomHorizontalFlip(),
                #                                 kornia.augmentation.RandomHue(p=0.5),
                #                                 kornia.augmentation.RandomSaturation(p=0.5)
                #                                 )

                # self.img_augment = nn.Sequential(
                #                                  kornia.augmentation.RandomHue(p=0.5),
                #                                  kornia.augmentation.RandomSaturation(p=0.5)
                #                                  )

        self.dwt = DWTForward(J=1, mode='zero', wave='db1').to(self.device)
        self.idwt = DWTInverse(mode="zero", wave="db1").to(self.device)

        self.lossDict = {}
        self.print_loss_dict = {}
        self.im_dict = {}
        self.val_im_dict = {}

    def train(self):
        # self.G_Net.encoder.train()
        # self.G_Net.decoder.train()
        self.G_Net.train()
        self.D_Net.train()

    def eval(self):
        self.G_Net.eval()


    def set_input(self,real_imgs,masks):
        self.mask = masks[:, 2:3, :, :]
        self.real_imgs = self.preprocess(real_imgs) #scale to -1 ~ 1
        masked_im = self.real_imgs * self.mask  # 0 for holes
        self.input = torch.cat((masked_im,self.mask),dim=1)

    def make_sample(self,imgs,masks):
        samples = self.teacher.forward(imgs,masks)
        return samples

    def forward(self,batch,count):
        self.count = count
        self.set_input(*batch)
        # if self.mode == 1:
        out,ws = self.G_Net(self.input)
        self.fake_imgs = out["img"]
        self.fake_freq = out["freq"][-1]
        self.comp_imgs = self.real_imgs * self.mask + self.fake_imgs * (1 - self.mask)
        # if self.opt.enable_teacher:
        self.teacher_out = self.make_sample(self.real_imgs,self.mask)
        self.real_imgs = self.teacher_out

    def backward_G(self):
        g_loss_list = []
        g_loss_name_list = []

        if self.opt.use_rec_loss:
            # if not self.opt.enable_teacher:
            #     rec_loss = self.opt.lambda_valid * l1_loss(self.real_imgs, self.fake_imgs,
            #                                                     self.mask)  # keep background unchanged
            # else:
            rec_loss = self.opt.lambda_valid * l1_loss(self.teacher_out,self.fake_imgs)

            g_loss_list.append(rec_loss)
            g_loss_name_list.append("rec_loss")

        if self.opt.use_rec_freq_loss:
            # if not self.opt.enable_teacher:
            #     # frequentcy reconstruct loss
            #     real_freq = self.img_to_dwt(self.real_imgs)
            #     mask_ = F.interpolate(self.mask, size=real_freq.size(-1), mode="nearest")
            #     rec_freq_loss = self.opt.lambda_hole * l1_loss(real_freq, self.fake_freq, mask_)
            # else:
                # frequentcy reconstruct loss
            real_freq = self.img_to_dwt(self.teacher_out)
            rec_freq_loss = self.opt.lambda_hole * l1_loss(real_freq, self.fake_freq)
            g_loss_list.append(rec_freq_loss)
            g_loss_name_list.append("rec_freq_loss")

        if self.opt.use_gan_loss:
            # if self.opt.D_input_type == "comp_img":
            #adversarial loss & feature matching loss with gradient penalty
            dis_comp, comp_d_feats = self.D_Net(self.comp_imgs)
            # gen_loss = Gen_loss_mask(dis_comp, (1 - self.mask), type=self.opt.gan_loss_type)
            gen_loss = Gen_loss(dis_comp, type=self.opt.gan_loss_type)
            # else:
            #     dis_fake, fake_d_feats = self.D_Net(self.fake_imgs)
            #     gen_loss = Gen_loss(dis_fake, type=self.opt.gan_loss_type)

            gen_loss = self.opt.lambda_gen * gen_loss
            g_loss_list.append(gen_loss)
            g_loss_name_list.append("gen_loss")

        if self.opt.use_perc_loss:
            # if self.opt.D_input_type == "comp_img":
            # if not self.opt.enable_teacher:
            #     perc_loss, *_ = self.lossNet(self.comp_imgs, self.real_imgs)
            # else:
            perc_loss, *_ = self.lossNet(self.comp_imgs, self.teacher_out)
            # else:
            #     if not self.opt.enable_teacher:
            #         perc_loss, *_ = self.lossNet(self.fake_imgs, self.real_imgs)
            #     else:
            #         perc_loss, *_ = self.lossNet(self.fake_imgs, self.teacher_out)

            perc_loss = self.opt.lambda_perc * perc_loss
            g_loss_list.append(perc_loss)
            g_loss_name_list.append("perc_loss")


        G_loss = 0.0
        for loss_name,loss in zip(g_loss_name_list,g_loss_list):
            self.lossDict[loss_name] = loss.item()
            G_loss += loss

        self.accelerator.backward(G_loss)

    def backward_D(self):
        if self.opt.gan_loss_type == 'R1':
            self.real_imgs.requires_grad = True

        dis_real, self.real_d_feats = self.D_Net(self.real_imgs)

        # if self.opt.D_input_type == "comp_img":
        # dis_comp, _ = self.D_Net(self.comp_imgs.detach())

        # dis_loss,r1_loss = Dis_loss_mask(dis_real, dis_comp, (1 - self.mask), real_bt=self.real_imgs,
        #                              type=self.opt.gan_loss_type,lambda_r1=self.opt.lambda_r1)
        # else:
        dis_comp, _ = self.D_Net(self.comp_imgs.detach())
        dis_loss,r1_loss = Dis_loss(dis_real, dis_comp, real_bt=self.real_imgs,
                                 type=self.opt.gan_loss_type,lambda_r1=self.opt.lambda_r1)

        self.lossDict['dis_loss'] = dis_loss.item()
        self.lossDict['r1_loss'] = r1_loss.item()

        self.accelerator.backward(dis_loss)


    def optimize_params(self):
        if self.opt.use_gan_loss:
            with self.accelerator.accumulate(self.D_Net):
                self.D_opt.zero_grad(set_to_none=True)
                self.backward_D()
                self.D_opt.step()

        with self.accelerator.accumulate(self.G_Net):
            self.G_opt.zero_grad(set_to_none=True)
            self.backward_G()
            self.G_opt.step()
            if self.opt.enable_ema:
                self.ema_G.update(self.G_Net.parameters())


        self.logging()

    def adjust_learning_rate(self, lr_in, min_lr, optimizer, epoch, lr_factor=0.95, warm_up=False, name='lr'):
        if not warm_up:
            lr = max(lr_in * lr_factor, float(min_lr))
        else:
            lr = max(lr_in * (epoch / int(self.opt.warm_up_epoch)), float(min_lr))

        print(f'Adjust learning rate to {lr:.5f}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        setattr(self, f'current_{name}', lr)

    @torch.no_grad()
    def validate(self,batch,count):
        self.val_count = count
        self.set_input(*batch)
        if self.mode == 1:
            unwrap_model = self.accelerator.unwrap_model(self.G_Net)
            unwrap_model = unwrap_model.to(self.device)
            out = unwrap_model(self.input)[0]['img']
            fake_imgs = self.real_imgs * self.mask + out * (1 - self.mask)
            fake_imgs = self.postprocess(fake_imgs)
            real_imgs = self.postprocess(self.real_imgs)
            masked_imgs = real_imgs * self.mask

        if self.opt.record_val_imgs:
            self.val_im_dict['fake_imgs'] = fake_imgs.cpu().detach()
            self.val_im_dict['real_imgs'] = real_imgs.cpu().detach()
            self.val_im_dict['masked_imgs'] = masked_imgs.cpu().detach()

        self.logging()

        return tensor2cv(real_imgs),tensor2cv(fake_imgs),tensor2cv(masked_imgs)

    def get_current_imgs(self):
        if self.mode == 1:
            self.im_dict['real_imgs'] = self.real_imgs.cpu().detach()
            self.im_dict['masked_imgs'] = (self.real_imgs * self.mask).cpu().detach()
            self.im_dict['fake_imgs'] = self.fake_imgs.cpu().detach()
            self.im_dict['comp_imgs'] = self.comp_imgs.cpu().detach()

    def logging(self):
        for lossName, lossValue in self.lossDict.items():
            self.recorder.add_scalar(lossName, lossValue, self.count)

        if self.print_loss_dict == {}:
            temp = {k:[] for k in self.lossDict.keys()}
            self.print_loss_dict.update(temp)
            self.print_loss_dict['r1_loss'] = []
        else:
            for k,v in self.lossDict.items():
                if k in self.print_loss_dict.keys():
                    self.print_loss_dict[k].append(v)


        if self.opt.record_training_imgs:
            if self.count % self.opt.save_im_step == 0:
                self.get_current_imgs()
                for im_name,im in self.im_dict.items():
                    im_grid = vutils.make_grid(im, normalize=False, scale_each=True)
                    self.recorder.add_image(im_name,im_grid,self.count)

        if self.opt.record_val_imgs:
            if self.count % self.opt.val_step == 0:
                for im_name, im in self.val_im_dict.items():
                    im_grid = vutils.make_grid(im, normalize=False, scale_each=True)
                    self.recorder.add_image(im_name, im_grid, self.val_count)

    def reduce_loss(self):
        for k, v in self.print_loss_dict.items():
            if len(v) != 0:
                self.print_loss_dict[k] = sum(v) / len(v)
            else:
                self.print_loss_dict[k] = 0.0


    #save validate imgs
    def save_results(self,val_real_ims,val_fake_ims,val_masked_ims=None):
        im_index = 0
        val_save_dir = os.path.join(self.val_saveDir, 'val_results')
        if os.path.exists((val_save_dir)):
            shutil.rmtree(val_save_dir)
        checkDir([val_save_dir])
        if self.mode == 1:
            for real_im, comp_im, masked_im in zip(val_real_ims, val_fake_ims, val_masked_ims):
                Image.fromarray(real_im).save(val_save_dir + '/{:0>5d}_im_truth.jpg'.format(im_index))
                Image.fromarray(comp_im).save(val_save_dir + '/{:0>5d}_im_out.jpg'.format(im_index))
                Image.fromarray(masked_im).save(val_save_dir + '/{:0>5d}_im_masked.jpg'.format(im_index))
                im_index += 1

    def load_from(self,model_path):
        model_state_dict = torch.load(model_path)
        new_state_dict = OrderedDict(
            {k.replace("generator.", ""): v for k, v in model_state_dict.items() if k.startswith("generator")})
        new_state_dict_ = OrderedDict(
            {k.replace("mapping_net.", ""): v for k, v in model_state_dict.items() if k.startswith("mapping_net")})
        self.G_Net.generator.load_state_dict(new_state_dict)
        self.G_Net.mapping_net.load_state_dict(new_state_dict_)

    def load(self):
        if self.mode == 1:
            self.load_network(self.saveDir, self.G_Net, load_last=self.opt.load_last,load_from_iter=self.opt.load_from_iter)
            self.load_network(self.saveDir + '/latest_dis.pth', self.D_Net, load_last=self.opt.load_last)

    # save checkpoint
    def save_network(self,loss_mean_val,val_type='default'):

        src_save_path = os.path.join(self.saveDir, f"last_G_{val_type}.pth")

        save_path = os.path.join(self.saveDir,
                "G-step={}_lr={}_{}_loss={}.pth".format(self.count+1, round(self.current_lr,6), val_type,loss_mean_val))
        dis_save_path = os.path.join(self.saveDir, 'latest_dis.pth')

        self.accelerator.print('saving network...')


        #work for ditributed training
        # if self.opt.acc_save:
        if self.mode  == 1:
            os.rename(src_save_path,save_path)
            unwrap_model = self.accelerator.unwrap_model(self.D_Net)
            self.accelerator.save(unwrap_model.state_dict(), dis_save_path)

        self.accelerator.print('saving network done. ')

    def normalize(self,t, range=(-1, 1)):
        t.clamp_(min=range[0], max=range[1])
        return t

    def preprocess(self, x):
        return x / 0.5 - 1.0

    def postprocess(self, x):
        return (x + 1.0) * 0.5

    def requires_grad(self,model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

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

class EncoderTrainer(BaseModel):
    def __init__(self,opt):
        super(EncoderTrainer, self).__init__(opt)
        self.count = 0
        self.opt = opt
        self.mode = opt.mode
        self.lossNet = ResNetPL(weights_path=opt.lossNetDir) #segmentation network for calculate percptual loss
        self.flops = None
        self.device = self.accelerator.device
        self.lossNet = self.lossNet.to(self.device)
        self.recorder = SummaryWriter(self.log_path)
        self.current_lr = opt.lr
        self.current_d_lr = opt.d_lr

        if self.mode == 1:
            # self.G_Net = MobileFill(input_nc=4,device=self.device,target_size=opt.targetSize)
            self.G_Net = StyleEncoder(config=[128, 256, 256, 512],input_nc=4, input_size=opt.targetSize)
            self.D_Net = MultidilatedNLayerDiscriminatorWithAtt(input_nc=3)
            # self.latent_D_Net = LatentCodesDiscriminator(style_dim=512, n_mlp=4)
            self.G_opt = torch.optim.AdamW(self.G_Net.parameters(), opt.lr,
                                           betas=(opt.beta_g_min, opt.beta_g_max))
            self.D_opt = torch.optim.AdamW(self.D_Net.parameters(), lr=opt.d_lr,
                                           betas=(opt.beta_d_min, opt.beta_d_max))

            # self.latent_D_opt = torch.optim.AdamW(self.latent_D_Net.parameters(), lr=opt.d_lr,
            #                                betas=(opt.beta_d_min, opt.beta_d_max))

            if opt.restore_training:
                self.load()

            if self.opt.enable_ema:
                self.ema_G = ExponentialMovingAverage(self.G_Net.parameters(), decay=0.995)
                self.ema_G.to(self.device)
                self.acc_args = [self.G_Net, self.D_Net, self.G_opt, self.D_opt]
            else:
                #args that should be prepared for accelerator
                self.acc_args = [self.G_Net, self.D_Net,self.G_opt, self.D_opt]


            self.teacher = StyleGANModel(model_path=opt.teacher_path, targetSize=opt.targetSize,device=self.device)

            self.img_aug = torch.nn.Sequential(
                                kornia.augmentation.RandomHorizontalFlip(),
                                kornia.augmentation.RandomAffine(
                                    translate=(0.1, 0.3),
                                    scale=(0.7, 1.2),
                                    degrees=(-20, 20)
                                ),
                                )

        self.lossDict = {}
        self.print_loss_dict = {}
        self.im_dict = {}
        self.val_im_dict = {}

    def train(self):
        self.G_Net.train()
        self.D_Net.train()

    def eval(self):
        self.G_Net.eval()


    def set_input(self,real_imgs,masks):
        self.mask = masks[:, 2:3, :, :]
        # self.real_imgs = self.preprocess(real_imgs) #scale to -1 ~ 1
        masked_im = real_imgs * self.mask  # 0 for holes
        self.input = torch.cat((masked_im,self.mask),dim=1)

    def make_sample(self,latents):
        samples = self.teacher.sample_with_latent(latents)
        return samples

    def forward(self,batch,count):
        self.count = count
        masks, noise = batch
        true_latents = self.teacher.model.get_latent(noise)
        true_latents = true_latents.unsqueeze(1).repeat(1, self.G_Net.n_latent, 1)
        with torch.no_grad():
            self.real_imgs = self.img_aug(self.make_sample(latents = true_latents))
        self.set_input(self.real_imgs,masks)
        fake_latents,self.deltas, _ = self.G_Net(self.input,noise)
        self.fake_imgs = self.make_sample(latents=fake_latents)


    def backward_G(self):
        g_loss_list = []
        g_loss_name_list = []

        if self.opt.use_rec_loss:
            rec_loss = self.opt.lambda_valid * l1_loss(self.real_imgs,self.fake_imgs)

            g_loss_list.append(rec_loss)
            g_loss_name_list.append("rec_loss")

        if self.opt.use_gan_loss:
            #adversarial loss & feature matching loss with gradient penalty
            dis_fake, fake_d_feats = self.D_Net(self.fake_imgs)
            gen_loss = Gen_loss(dis_fake, type=self.opt.gan_loss_type)

            gen_loss = self.opt.lambda_gen * gen_loss
            g_loss_list.append(gen_loss)
            g_loss_name_list.append("gen_loss")

        if self.opt.use_perc_loss:
            perc_loss, *_ = self.lossNet(self.fake_imgs, self.real_imgs)

            perc_loss = self.opt.lambda_perc * perc_loss
            g_loss_list.append(perc_loss)
            g_loss_name_list.append("perc_loss")

        if self.opt.use_delta_loss:
            delta_loss = self.opt.lambda_delta * torch.norm(self.deltas, 2, dim=1).mean()
            g_loss_list.append(delta_loss)
            g_loss_name_list.append("delta_loss")


        G_loss = 0.0
        for loss_name,loss in zip(g_loss_name_list,g_loss_list):
            self.lossDict[loss_name] = loss.item()
            G_loss += loss

        self.accelerator.backward(G_loss)

    def backward_D(self):
        if self.opt.gan_loss_type == 'R1':
            self.real_imgs.requires_grad = True

        dis_real, self.real_d_feats = self.D_Net(self.real_imgs)

        dis_fake, _ = self.D_Net(self.fake_imgs.detach())
        dis_loss, r1_loss = Dis_loss(dis_real, dis_fake, real_bt=self.real_imgs,
                                     type=self.opt.gan_loss_type, lambda_r1=self.opt.lambda_r1)

        self.lossDict['dis_loss'] = dis_loss.item()
        self.lossDict['r1_loss'] = r1_loss.item()

        self.accelerator.backward(dis_loss)


    def optimize_params(self):
        if self.opt.use_gan_loss:
            with self.accelerator.accumulate(self.D_Net):
                self.D_opt.zero_grad(set_to_none=True)
                self.backward_D()
                self.D_opt.step()

        with self.accelerator.accumulate(self.G_Net):
            self.G_opt.zero_grad(set_to_none=True)
            self.backward_G()
            self.G_opt.step()
            if self.opt.enable_ema:
                self.ema_G.update(self.G_Net.parameters())


        self.logging()

    def adjust_learning_rate(self, lr_in, min_lr, optimizer, epoch, lr_factor=0.95, warm_up=False, name='lr'):
        if not warm_up:
            lr = max(lr_in * lr_factor, float(min_lr))
        else:
            lr = max(lr_in * (epoch / int(self.opt.warm_up_epoch)), float(min_lr))

        print(f'Adjust learning rate to {lr:.5f}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        setattr(self, f'current_{name}', lr)

    @torch.no_grad()
    def validate(self,batch,count):
        self.val_count = count
        unwrap_model = self.accelerator.unwrap_model(self.G_Net)

        masks, noise = batch
        true_latents = self.teacher.model.get_latent(noise)
        true_latents = true_latents.unsqueeze(1).repeat(1, self.G_Net.n_latent, 1)
        with torch.no_grad():
            real_imgs = self.make_sample(latents=true_latents)
        self.set_input(real_imgs, masks)
        fake_latents, deltas, _ = unwrap_model(self.input, noise)
        fake_imgs = self.make_sample(latents = fake_latents)

        fake_imgs = real_imgs * self.mask + fake_imgs * (1 - self.mask)
        fake_imgs = self.postprocess(fake_imgs)
        real_imgs = self.postprocess(real_imgs)
        masked_imgs = real_imgs * self.mask

        if self.opt.record_val_imgs:
            self.val_im_dict['fake_imgs'] = fake_imgs.cpu().detach()
            self.val_im_dict['real_imgs'] = real_imgs.cpu().detach()
            self.val_im_dict['masked_imgs'] = masked_imgs.cpu().detach()

        self.logging()

        return tensor2cv(real_imgs),tensor2cv(fake_imgs),tensor2cv(masked_imgs)

    def get_current_imgs(self):
        self.im_dict['real_imgs'] = self.real_imgs.cpu().detach()
        self.im_dict['masked_imgs'] = (self.real_imgs * self.mask).cpu().detach()
        self.im_dict['fake_imgs'] = self.fake_imgs.cpu().detach()

    def logging(self):
        for lossName, lossValue in self.lossDict.items():
            self.recorder.add_scalar(lossName, lossValue, self.count)

        if self.print_loss_dict == {}:
            temp = {k:[] for k in self.lossDict.keys()}
            self.print_loss_dict.update(temp)
            self.print_loss_dict['r1_loss'] = []
        else:
            for k,v in self.lossDict.items():
                if k in self.print_loss_dict.keys():
                    self.print_loss_dict[k].append(v)


        if self.opt.record_training_imgs:
            if self.count % self.opt.save_im_step == 0:
                self.get_current_imgs()
                for im_name,im in self.im_dict.items():
                    im_grid = vutils.make_grid(im, normalize=False, scale_each=True)
                    self.recorder.add_image(im_name,im_grid,self.count)

        if self.opt.record_val_imgs:
            if self.count % self.opt.val_step == 0:
                for im_name, im in self.val_im_dict.items():
                    im_grid = vutils.make_grid(im, normalize=False, scale_each=True)
                    self.recorder.add_image(im_name, im_grid, self.val_count)

    def reduce_loss(self):
        for k, v in self.print_loss_dict.items():
            if len(v) != 0:
                self.print_loss_dict[k] = sum(v) / len(v)
            else:
                self.print_loss_dict[k] = 0.0


    #save validate imgs
    def save_results(self,val_real_ims,val_fake_ims,val_masked_ims=None):
        im_index = 0
        val_save_dir = os.path.join(self.val_saveDir, 'val_results')
        if os.path.exists((val_save_dir)):
            shutil.rmtree(val_save_dir)
        checkDir([val_save_dir])
        if self.mode == 1:
            for real_im, comp_im, masked_im in zip(val_real_ims, val_fake_ims, val_masked_ims):
                Image.fromarray(real_im).save(val_save_dir + '/{:0>5d}_im_truth.jpg'.format(im_index))
                Image.fromarray(comp_im).save(val_save_dir + '/{:0>5d}_im_out.jpg'.format(im_index))
                Image.fromarray(masked_im).save(val_save_dir + '/{:0>5d}_im_masked.jpg'.format(im_index))
                im_index += 1

    def load_from(self,model_path):
        model_state_dict = torch.load(model_path)
        new_state_dict = OrderedDict(
            {k.replace("generator.", ""): v for k, v in model_state_dict.items() if k.startswith("generator")})
        new_state_dict_ = OrderedDict(
            {k.replace("mapping_net.", ""): v for k, v in model_state_dict.items() if k.startswith("mapping_net")})
        self.G_Net.generator.load_state_dict(new_state_dict)
        self.G_Net.mapping_net.load_state_dict(new_state_dict_)

    def load(self):
        if self.mode == 1:
            self.load_network(self.saveDir, self.G_Net, load_last=self.opt.load_last,load_from_iter=self.opt.load_from_iter)
            self.load_network(self.saveDir + '/latest_dis.pth', self.D_Net, load_last=self.opt.load_last)

    # save checkpoint
    def save_network(self,loss_mean_val,val_type='default'):

        src_save_path = os.path.join(self.saveDir, f"last_G_{val_type}.pth")

        save_path = os.path.join(self.saveDir,
                "G-step={}_lr={}_{}_loss={}.pth".format(self.count+1, round(self.current_lr,6), val_type,loss_mean_val))
        dis_save_path = os.path.join(self.saveDir, 'latest_dis.pth')

        self.accelerator.print('saving network...')


        #work for ditributed training
        # if self.opt.acc_save:
        if self.mode  == 1:
            os.rename(src_save_path,save_path)
            unwrap_model = self.accelerator.unwrap_model(self.D_Net)
            self.accelerator.save(unwrap_model.state_dict(), dis_save_path)

        self.accelerator.print('saving network done. ')

    def normalize(self,t, range=(-1, 1)):
        t.clamp_(min=range[0], max=range[1])
        return t

    def preprocess(self, x):
        return x / 0.5 - 1.0

    def postprocess(self, x):
        return (x + 1.0) * 0.5









