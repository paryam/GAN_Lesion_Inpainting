import os
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import InpaintGenerator, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss


class BaseModel(nn.Module):
    def __init__(self, name, run_path, config):
        super(BaseModel, self).__init__()

        self.config = config
        self.iteration = 0
        self.name = name

        self.gen_weights_path = os.path.join(run_path, name + '_gen')
        self.dis_weights_path = os.path.join(run_path, name + '_dis')

    def load(self, iteration):
        if os.path.exists(self.gen_weights_path + "-{0}.pth".format(iteration)):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path + "-{0}.pth".format(iteration))
            else:
                data = torch.load(self.gen_weights_path + "-{0}.pth".format(iteration),
                                  map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path + "-{0}.pth".format(iteration)):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path + "-{0}.pth".format(iteration))
            else:
                data = torch.load(self.dis_weights_path + "-{0}.pth".format(iteration),
                                  map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path + "-{0}.pth".format(self.iteration))

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path + "-{0}.pth".format(self.iteration))


class InpaintingModel(BaseModel):
    def __init__(self, config, run_path, name="InpaintingModel"):
        super(InpaintingModel, self).__init__(name, run_path, config)

        generator = InpaintGenerator(use_spectral_norm=True)
        discriminator = Discriminator(in_channels=1, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator , config.GPU)

        l1_loss = nn.L1Loss()
        mse_loss = nn.MSELoss()
        perceptual_loss = PerceptualLoss(native=config.NATIVE)
        style_loss = StyleLoss(native=config.NATIVE)
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('mse_loss', mse_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )
        self.gen_scheduler = optim.lr_scheduler.ExponentialLR(self.gen_optimizer, gamma=config.EXP_DECAY)

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )
        self.dis_scheduler = optim.lr_scheduler.ExponentialLR(self.dis_optimizer, gamma=config.EXP_DECAY)

    def process(self, images, edges, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, dis_real_feat = self.discriminator(dis_input_real)
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss

        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss

        # gen_mse_loss = self.mse_loss(outputs, images) * self.config.MSE_LOSS_WEIGHT / torch.mean(masks)
        # gen_loss += gen_mse_loss

        # generator perceptual loss
        # outputs_3 = torch.cat([outputs, outputs, outputs], dim=1)
        # images_3 = torch.cat([images, images, images], dim=1)
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss

        # generator style loss
        # gen_style_loss = self.style_loss(outputs_3 * masks, images_3 * masks)
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            # ("l_mse", gen_mse_loss.item()),
            ("l_fm", gen_fm_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        images_masked = (images * (1 - masks).float())
        inputs = torch.cat((images_masked, edges), dim=1)
        outputs = self.generator(inputs)
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        gen_loss.backward()

        self.dis_optimizer.step()
        self.gen_optimizer.step()
