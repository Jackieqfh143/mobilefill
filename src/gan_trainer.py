import os
import shutil
import torch
import torch.nn as nn
from src.models.baseModel import BaseModel
from src.evaluate.loss import ResNetPL,l1_loss,Gen_loss,Dis_loss
from src.utils.util import checkDir,tensor2cv
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from PIL import Image
from torch_ema import ExponentialMovingAverage
from pytorch_wavelets import DWTInverse, DWTForward
from src.models.mobileFill import MobileFill,MobileFill_v2
from src.models.discriminator import MultidilatedNLayerDiscriminatorWithAtt, MultidilatedNLayerDiscriminatorWithAtt_v2,\
    MultidilatedNLayerDiscriminatorWithAtt_UNet,UNetDiscriminator,StyleGAN_Discriminator,EESPDiscriminator
from src.models.stylegan import StyleGANModel
from collections import OrderedDict


class GAN_trainer(BaseModel):
    def __init__(self,opt):
        super(GAN_trainer, self).__init__(opt)
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
            teacher_state_dict = torch.load(opt.teacher_path)["g_ema"]
            mapping_state_dict = OrderedDict({k.replace("style", "layers"): v for k, v in teacher_state_dict.items() if k.startswith("style")})
            self.G_Net = MobileFill(input_nc=4,device=self.device,target_size=opt.targetSize)
            self.G_Net.mapping_net.load_state_dict(mapping_state_dict)
            self.G_Net.mapping_net.eval().requires_grad_(False)
            self.G_Net = self.G_Net.to(self.device)
            self.D_Net = MultidilatedNLayerDiscriminatorWithAtt(input_nc=3)
            # self.D_Net = StyleGAN_Discriminator(size=opt.targetSize,channels_in=3)

            self.G_opt = torch.optim.AdamW(self.G_Net.parameters(), opt.lr,
                                           betas=(opt.beta_g_min, opt.beta_g_max))
            self.D_opt = torch.optim.AdamW(self.D_Net.parameters(), lr=opt.d_lr,
                                           betas=(opt.beta_d_min, opt.beta_d_max))

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
                self.teacher = StyleGANModel(model_path=opt.teacher_path,device=self.device,targetSize=self.opt.targetSize)

        self.dwt = DWTForward(J=1, mode='zero', wave='db1').to(self.device)
        self.idwt = DWTInverse(mode="zero", wave="db1").to(self.device)

        self.lossDict = {}
        self.print_loss_dict = {}
        self.im_dict = {}
        self.val_im_dict = {}

    def train(self):
        self.G_Net.generator.train()
        self.G_Net.encoder.train()
        self.D_Net.train()

    def eval(self):
        self.G_Net.eval()

    def make_sample(self,sample_z):
        samples = self.teacher.sample_with_latent(sample_z)
        return samples

    def forward(self,batch,count):
        self.count = count
        self.real_imgs, latents = self.make_sample(batch)
        self.fake_imgs = self.forward_G(batch)

    def forward_G(self,noise,style = None):
        return self.G_Net.gan_forward(noise,style)

    def backward_G(self):
        g_loss_list = []
        g_loss_name_list = []

        if self.opt.use_rec_loss:
            rec_loss = self.opt.lambda_valid * l1_loss(self.real_imgs,self.fake_imgs)
            g_loss_list.append(rec_loss)
            g_loss_name_list.append("rec_loss")

        if self.opt.use_rec_freq_loss:
            # frequentcy reconstruct loss
            fake_freq = self.img_to_dwt(self.fake_imgs)
            real_freq = self.img_to_dwt(self.real_imgs)
            rec_freq_loss = self.opt.lambda_hole * l1_loss(real_freq, fake_freq)
            g_loss_list.append(rec_freq_loss)
            g_loss_name_list.append("rec_freq_loss")

        if self.opt.use_gan_loss:
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
        dis_loss,r1_loss = Dis_loss(dis_real, dis_fake, real_bt=self.real_imgs,
                                 type=self.opt.gan_loss_type,lambda_r1=self.opt.lambda_r1)

        self.lossDict['dis_loss'] = dis_loss.item()
        self.lossDict['r1_loss'] = r1_loss.item()

        self.accelerator.backward(dis_loss)

    def optimize_params(self):
        if self.opt.use_gan_loss:
            with self.accelerator.accumulate(self.D_Net):
                self.D_opt.zero_grad()
                self.backward_D()
                self.D_opt.step()

        with self.accelerator.accumulate(self.G_Net):
            self.G_opt.zero_grad()
            self.backward_G()
            if self.opt.use_grad_norm:
                # gradient clip
                self.accelerator.clip_grad_norm_(parameters=self.G_Net.parameters(),
                                            max_norm=self.opt.max_grad_norm,
                                            norm_type=self.opt.grad_norm_type)
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
        unwrap_model = unwrap_model.to(self.device)
        fake_imgs = unwrap_model.gan_forward(batch)
        real_imgs,_ = self.teacher.sample(batch)
        fake_imgs = self.postprocess(fake_imgs)
        real_imgs = self.postprocess(real_imgs)

        if self.opt.record_val_imgs:
            self.val_im_dict['fake_imgs'] = fake_imgs.cpu().detach()
            self.val_im_dict['real_imgs'] = real_imgs.cpu().detach()

        self.logging()

        return tensor2cv(real_imgs),tensor2cv(fake_imgs)

    def get_current_imgs(self):
        if self.mode == 1:
            self.im_dict['real_imgs'] = self.real_imgs.cpu().detach()
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
    def save_results(self,val_real_ims,val_fake_ims):
        im_index = 0
        val_save_dir = os.path.join(self.val_saveDir, 'val_results')
        if os.path.exists((val_save_dir)):
            shutil.rmtree(val_save_dir)
        checkDir([val_save_dir])
        if self.mode == 1:
            for real_im, fake_im in zip(val_real_ims, val_fake_ims):
                Image.fromarray(real_im).save(val_save_dir + '/{:0>5d}_im_truth.jpg'.format(im_index))
                Image.fromarray(fake_im).save(val_save_dir + '/{:0>5d}_im_out.jpg'.format(im_index))
                im_index += 1

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




