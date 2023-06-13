import os
import shutil
import torch
from src.models.baseModel import BaseModel
from src.evaluate.loss import ResNetPL,l1_loss,Dis_loss_mask,Gen_loss,Dis_loss
from src.utils.util import checkDir,tensor2cv
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from PIL import Image
from torch_ema import ExponentialMovingAverage
from src.models.main import  MobileFill_v3
from src.models.discriminator import MultidilatedNLayerDiscriminatorWithAtt
from collections import OrderedDict
from src.models.stylegan import StyleGANModel
import kornia


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

        self.G_Net = MobileFill_v3(input_nc=4,device=self.device,target_size=opt.targetSize)
        self.G_Net.eval().requires_grad_(False)
        self.D_Net = MultidilatedNLayerDiscriminatorWithAtt(input_nc=3)
        self.G_opt = torch.optim.AdamW(self.G_Net.parameters(), opt.lr,
                                       betas=(opt.beta_g_min, opt.beta_g_max))
        self.D_opt = torch.optim.AdamW(self.D_Net.parameters(), lr=opt.d_lr,
                                       betas=(opt.beta_d_min, opt.beta_d_max))

        self.G_Net.encoder.load_state_dict(torch.load(self.opt.model_path))


        if opt.restore_training:
            self.load()

        if self.opt.enable_ema:
            self.ema_G = ExponentialMovingAverage(self.G_Net.parameters(), decay=0.995)
            self.ema_G.to(self.device)


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
        if self.mode == 1:
            self.G_Net.encoder.train()

        elif self.mode == 2:
            self.G_Net.generator.train()

        else:
            self.G_Net.refine_encoder.train()
            self.G_Net.refine_decoder.train()

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

    """
    mode1: train the latent encoder 
    mode2: train the generator
    mode3: train the refiner
    """

    def forward(self,batch,count):
        self.count = count
        masks, noise = batch
        self.mask = masks[:, 2:3, :, :]
        true_latent = self.teacher.model.get_latent(noise)
        true_latents = true_latent.unsqueeze(1).repeat(1, self.G_Net.latent_num, 1)
        with torch.no_grad():
            self.real_imgs = self.img_aug(self.make_sample(latents = true_latents))

        fake_latents, self.deltas, _ = self.G_Net.get_latent(self.real_imgs,masks,noise)
        gan_fake_imgs = self.make_sample(latents=fake_latents)
        if self.mode == 1:
            self.fake_imgs = gan_fake_imgs
        elif self.mode == 2:
            self.fake_imgs = self.G_Net.generator(fake_latents)
        else:
            merged_x = self.real_imgs * self.mask + gan_fake_imgs * (1 - self.mask)
            self.fake_imgs = self.G_Net.refine(merged_x)


    def backward_G(self):
        g_loss_list = []
        g_loss_name_list = []

        if self.opt.use_rec_loss:
            if self.mode == 3:
                # keep background unchanged
                rec_loss = self.opt.lambda_valid * l1_loss(self.real_imgs,self.fake_imgs,self.mask)
            else:
                rec_loss = self.opt.lambda_valid * l1_loss(self.real_imgs, self.fake_imgs)

            g_loss_list.append(rec_loss)
            g_loss_name_list.append("rec_loss")

        if self.opt.use_gan_loss:
            #adversarial loss & feature matching loss with gradient penalty
            dis_fake, fake_d_feats = self.D_Net(self.fake_imgs)
            gen_loss = self.opt.lambda_gen * Gen_loss(dis_fake, type=self.opt.gan_loss_type)
            g_loss_list.append(gen_loss)
            g_loss_name_list.append("gen_loss")

        if self.opt.use_perc_loss:
            perc_loss, *_ = self.lossNet(self.fake_imgs, self.real_imgs)
            perc_loss = self.opt.lambda_perc * perc_loss
            g_loss_list.append(perc_loss)
            g_loss_name_list.append("perc_loss")

        if self.opt.use_delta_loss and self.mode != 3:
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
        if self.mode == 3:
            dis_loss, r1_loss = Dis_loss_mask(dis_real, dis_fake, (1 - self.mask), real_bt=self.real_imgs,
                                              type=self.opt.gan_loss_type, lambda_r1=self.opt.lambda_r1)
        else:
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
        unwrap_model = self.accelerator.unwrap_model(self.G_Net)
        self.val_count = count
        if self.mode == 3:
            real_imgs, masks = batch
            noise = torch.randn(real_imgs.size(0),self.teacher.model.style_dim).to(self.device)
        else:
            masks, noise = batch
            true_latent = self.teacher.model.get_latent(noise)
            true_latents = true_latent.unsqueeze(1).repeat(1, self.G_Net.latent_num, 1)
            real_imgs = self.make_sample(latents=true_latents)

        fake_latents, *_ = unwrap_model.get_latent(real_imgs, masks, noise)
        if self.mode == 1:
            fake_imgs = self.make_sample(latents=fake_latents)
        elif self.mode == 2:
            fake_imgs = unwrap_model.generator(fake_latents)
        else:
            # fake_imgs = unwrap_model.generator(fake_latents)
            gan_fake_imgs = self.make_sample(latents=fake_latents)
            merged_x = real_imgs * masks + gan_fake_imgs * (1 - masks)
            fake_imgs = unwrap_model.refine(merged_x)
            if self.opt.record_val_imgs and self.opt.debug:
                self.val_im_dict['merged_x'] = self.postprocess(merged_x).cpu().detach()

        fake_imgs = self.postprocess(fake_imgs)
        real_imgs = self.postprocess(real_imgs)
        masked_imgs = real_imgs * masks

        if self.opt.record_val_imgs:
            self.val_im_dict['real_imgs'] = real_imgs.cpu().detach()
            self.val_im_dict['fake_imgs'] = fake_imgs.cpu().detach()
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
        self.load_network(self.saveDir, self.G_Net, load_last=self.opt.load_last,load_from_iter=self.opt.load_from_iter)
        self.load_network(self.saveDir + '/latest_dis.pth', self.D_Net, load_last=self.opt.load_last)

    # save checkpoint
    def save_network(self,loss_mean_val,val_type='default'):

        src_save_path = os.path.join(self.saveDir, f"last_G_{val_type}.pth")

        save_path = os.path.join(self.saveDir,
                "G-step={}_lr={}_{}_loss={}.pth".format(self.count+1, round(self.current_lr,6), val_type,loss_mean_val))
        dis_save_path = os.path.join(self.saveDir, 'latest_dis.pth')
        dis_latent_save_path = os.path.join(self.saveDir, 'latest_latent_dis.pth')

        self.accelerator.print('saving network...')


        #work for ditributed training
        # if self.opt.acc_save:
        os.rename(src_save_path,save_path)
        unwrap_model = self.accelerator.unwrap_model(self.D_Net)
        self.accelerator.save(unwrap_model.state_dict(), dis_save_path)

        # unwrap_model = self.accelerator.unwrap_model(self.latent_D_Net)
        # self.accelerator.save(unwrap_model.state_dict(), dis_latent_save_path)

        self.accelerator.print('saving network done. ')

    def normalize(self,t, range=(-1, 1)):
        t.clamp_(min=range[0], max=range[1])
        return t

    def preprocess(self, x):
        return x / 0.5 - 1.0

    def postprocess(self, x):
        return (x + 1.0) * 0.5









