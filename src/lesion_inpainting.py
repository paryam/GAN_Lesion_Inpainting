import os

import copy
from collections import OrderedDict

import numpy as np
import SimpleITK as sitk
import torch
import torchvision
from itertools import product

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .dataset import Dataset
from .models import InpaintingModel
from .models_3D import InpaintingModel3D
from .utils import Progbar, create_dir, imsave, create_train_test_val_list, pad_and_or_crop
from .metrics import PSNR

torch.autograd.set_detect_anomaly(True)

class GANLesionInpainting():
    def __init__(self, config):
        self.config = config
        self.debug = False

        self.psnr = PSNR(255.0).to(config.DEVICE)

        if config.XFM_FLIST:
            images, lesions, brain_masks, xfms, tissue_masks = create_train_test_val_list(config)
        else:
            images, lesions, brain_masks, tissue_masks = create_train_test_val_list(config)
            xfms = None
        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(
                config, img_list=images[2][self.config.TEST_START_IND:self.config.TEST_END_IND],
                lesion_list=lesions[2][self.config.TEST_START_IND:self.config.TEST_END_IND],
                brain_mask_list=brain_masks[2][self.config.TEST_START_IND:self.config.TEST_END_IND],
                external_mask_list=config.TEST_MASK_FLIST, augment=False, training=False)
        elif self.config.MODE == 3:
            self.infer_dataset = Dataset(
                config, img_list=[self.config.input_image],
                lesion_list=[self.config.input_lesion],
                brain_mask_list=[self.config.brain_mask],
                tissue_mask_list=[self.config.tissue_mask],
                xfms=[self.config.xfm],
                augment=False, training=False, inference=True)
        else:
            self.train_dataset = Dataset(
                config, img_list=images[0], lesion_list=lesions[0], brain_mask_list=brain_masks[0],
                xfms=xfms[0], tissue_mask_list=tissue_masks[0], augment=True, training=True)
            self.val_dataset = Dataset(
                config, img_list=images[1], lesion_list=lesions[1], brain_mask_list=brain_masks[1],
                xfms=xfms[1], tissue_mask_list=tissue_masks[1], augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        parameters = OrderedDict(
            LR = [0.00005],  # [0.00005, 0.0001],
            EXP_DECAY = [0.98],
            D2G_LR = [0.1],
            L1_LOSS_WEIGHT = [1],  # [20, 10, 5],
            FM_LOSS_WEIGHT = [10],  # [20, 50, 10],
            STYLE_LOSS_WEIGHT = [50],       # style loss weight
            CONTENT_LOSS_WEIGHT = [1],  # [1, 0.1],        # perceptual loss weight
            INPAINT_ADV_LOSS_WEIGHT = [0.5],  # [1, 0.5, 0.1], # [0.5. 0.1]
        )

        self.param_values = [v for v in parameters.values()]

    def load(self):
        self.inpaint_model.load(self.config.iteration)

    def save(self):
        self.inpaint_model.save()

    def train(self):
        for r_id, (lr, exp_decay, dtg_lr, l1_loss_w, fm_loss_w, sty_loss_w, cnt_loss_w, inpaint_adv_loss_w) in enumerate(product(*self.param_values)):
            print(lr, exp_decay, dtg_lr, l1_loss_w, fm_loss_w, sty_loss_w, cnt_loss_w, inpaint_adv_loss_w )
            if self.config.run_id:
                run_id = self.config.run_id
            else:
                run_id = r_id + 1

            print("run id:", run_id)
            self.config.LR = lr
            self.config.EXP_DECAY = exp_decay
            self.config.D2G_LR = dtg_lr
            self.config.L1_LOSS_WEIGHT = l1_loss_w
            # self.config.MSE_LOSS_WEIGHT = mse_loss_w
            self.config.FM_LOSS_WEIGHT = fm_loss_w
            self.config.STYLE_LOSS_WEIGHT = sty_loss_w
            self.config.CONTENT_LOSS_WEIGHT = cnt_loss_w
            self.config.INPAINT_ADV_LOSS_WEIGHT = inpaint_adv_loss_w

            self.run_path = os.path.join(self.config.PATH, "Run-{0}".format(run_id))
            if os.path.exists(self.run_path):
                continue

            self.model_name = "GAN-InPainting-RunID-{0}".format(run_id)
            if self.config.model_3D:
                self.inpaint_model = InpaintingModel3D(
                    self.config, run_path=self.run_path, name=self.model_name).to(self.config.DEVICE)
            else:
                self.inpaint_model = InpaintingModel(
                    self.config, run_path=self.run_path, name=self.model_name).to(self.config.DEVICE)
            if not os.path.isdir(os.path.join(self.run_path)):
                os.makedirs(os.path.join(self.run_path))
            self.samples_path = os.path.join(self.run_path, 'samples')
            self.results_path = os.path.join(self.run_path, 'results')
            self.tensorboard_path = os.path.join(self.config.TENSORBOARD_DIR, 'runs')

            with open(os.path.join(self.run_path, "params.txt"), "w") as f:
                comment=f' lr = {lr} l1_loss_w = {l1_loss_w} fm_loss_w = {fm_loss_w} sty_loss_w = {sty_loss_w}' \
                        f'cnt_loss_w = {cnt_loss_w} inpaint_adv_loss_w = {inpaint_adv_loss_w}'
                f.write(comment)

            if self.config.RESULTS is not None:
                self.results_path = os.path.join(self.config.RESULTS)

            if self.config.DEBUG is not None and self.config.DEBUG != 0:
                self.debug = True

            self.log_file = os.path.join(self.run_path, 'log_' + self.model_name + '.dat')
            self.writer = SummaryWriter(
                # log_dir=self.tensorboard_path,
                comment=f' lr = {lr} l1_loss_w = {l1_loss_w} fm_loss_w = {fm_loss_w} sty_loss_w = {sty_loss_w}'
                        f'cnt_loss_w = {cnt_loss_w} inpaint_adv_loss_w = {inpaint_adv_loss_w}')

            self.load()
            train_values, val_values = self.train_ind(run_id)

            self.writer.add_hparams(
                {"lr": lr, "l1_loss_w": l1_loss_w, "fm_loss_w": fm_loss_w, "sty_loss_w": sty_loss_w,
                 "cnt_loss_w": cnt_loss_w, "inpaint_adv_loss_w": inpaint_adv_loss_w},
                {
                    "train_psnr" : train_values[0],
                    "train_mae" : train_values[1],
                    "train_gen_loss" : train_values[2],
                    "train_dis_loss" : train_values[3],
                    "val_psnr" : val_values[0],
                    "val_mae" : val_values[1],
                    "val_gen_loss" : val_values[2],
                    "val_dis_loss" : val_values[3],
                },
            )
            self.writer.close()

    def train_ind(self, run_id):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=min(self.config.BATCH_SIZE, 16),
            drop_last=True,
            shuffle=True
        )

        images, edges, masks = next(iter(train_loader))
        images, edges, masks = self.cuda(images, edges, masks)
        self.writer.add_graph(self.inpaint_model, (images, edges, masks,))
        self.writer.close()

        epoch = 0
        keep_training = True
        max_iteration = int(float(self.config.MAX_ITERS))
        total = len(self.train_dataset)
        epoch_psnr = 0
        epoch_mae = 0
        epoch_gen_loss = 0
        epoch_dis_loss= 0

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        prv_val_loss = np.Inf
        while keep_training:
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)
            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            running_gen_loss = 0
            running_dis_loss = 0
            running_psnr = 0
            running_mae = 0
            for items in train_loader:
                self.inpaint_model.train()

                images, edges, masks = self.cuda(*items)
                # train
                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

                # backward
                self.inpaint_model.backward(gen_loss, dis_loss)
                iteration = self.inpaint_model.iteration

                running_gen_loss += float(gen_loss) * self.config.BATCH_SIZE
                running_dis_loss += float(dis_loss) * self.config.BATCH_SIZE
                running_psnr += float(psnr) * self.config.BATCH_SIZE
                running_mae += float(mae) * self.config.BATCH_SIZE

                if iteration >= max_iteration or epoch >= self.config.MAX_EPOCH:
                    keep_training = False

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at end of epoch
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

            # sample, evaluate and save model at end of epoch
            with torch.no_grad():
                self.sample(run_id=run_id)
                val_gen_loss, val_dis_loss, val_psnr, val_mae = self.eval(epoch=epoch)

            if (val_gen_loss < prv_val_loss) or (keep_training is False):
                prv_val_loss = val_gen_loss
                self.save()

            epoch_gen_loss = running_gen_loss / len(self.train_dataset)
            epoch_dis_loss = running_dis_loss / len(self.train_dataset)
            epoch_psnr = running_psnr / len(self.train_dataset)
            epoch_mae = running_mae / len(self.train_dataset)

            self.writer.add_scalar("Metrics/train-psnr", epoch_psnr, epoch)
            self.writer.add_scalar("Metrics/train-mae", epoch_mae, epoch)
            self.writer.add_scalar("Loss/train-gen", epoch_gen_loss, epoch)
            self.writer.add_scalar("Loss/train-dis", epoch_dis_loss, epoch)

            self.writer.add_scalar(
                "LearningRate/LR-gen", np.array(self.inpaint_model.gen_scheduler.get_last_lr()), epoch)
            self.writer.add_scalar(
                "LearningRate/LR-dis", np.array(self.inpaint_model.dis_scheduler.get_last_lr()), epoch)
            if iteration > 28000:
                self.inpaint_model.gen_scheduler.step()
                self.inpaint_model.dis_scheduler.step()

        print('\nEnd training....')
        return [epoch_psnr, epoch_mae, epoch_gen_loss, epoch_dis_loss], [val_psnr, val_mae, val_gen_loss, val_dis_loss]

    def eval(self, epoch):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            num_workers=min(self.config.VAL_BATCH_SIZE, 16),
            batch_size=self.config.VAL_BATCH_SIZE,
            drop_last=True
        )

        total = len(self.val_dataset)

        self.inpaint_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        running_gen_loss = 0
        running_dis_loss = 0
        running_psnr = 0
        running_mae = 0
        for items in val_loader:
            iteration += 1
            images, edges, masks = self.cuda(*items)

            # eval
            outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

            # metrics
            psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
            mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
            logs.append(('psnr', psnr.item()))
            logs.append(('mae', mae.item()))

            running_gen_loss += float(gen_loss) * self.config.VAL_BATCH_SIZE
            running_dis_loss += float(dis_loss) * self.config.VAL_BATCH_SIZE
            running_psnr += float(psnr) * self.config.VAL_BATCH_SIZE
            running_mae += float(mae) * self.config.VAL_BATCH_SIZE

            logs = [("it", iteration), ] + logs
            progbar.add(len(images), values=logs)

        val_gen_loss = running_gen_loss / len(self.val_dataset)
        val_dis_loss = running_dis_loss / len(self.val_dataset)
        val_psnr = running_psnr / len(self.val_dataset)
        val_mae = running_mae / len(self.val_dataset)

        self.writer.add_scalar("Metrics/val-psnr", val_psnr, epoch)
        self.writer.add_scalar("Metrics/val-mae", val_mae, epoch)
        self.writer.add_scalar("Loss/val-gen", val_gen_loss, epoch)
        self.writer.add_scalar("Loss/val-dis", val_dis_loss, epoch)

        return val_gen_loss, val_dis_loss, val_psnr, val_mae

    def predict(self):
        run_id = self.config.run_id
        self.model_name = "GAN-InPainting-RunID-{0}".format(run_id)

        self.model_name = "GAN-InPainting-RunID-{0}".format(run_id)
        self.run_path = os.path.join(self.config.PATH, "Run-{0}".format(run_id))
        if self.config.model_3D:
            self.inpaint_model = InpaintingModel3D(
                self.config, run_path=self.run_path, name=self.model_name).to(self.config.DEVICE)
        else:
            self.inpaint_model = InpaintingModel(
                self.config, run_path=self.run_path, name=self.model_name).to(self.config.DEVICE)

        if self.config.outdir:
            self.results_path = self.config.outdir
        else:
            self.results_path = os.path.join(self.run_path, "results")
        if self.config.DEBUG is not None and self.config.DEBUG != 0:
            self.debug = True

        self.load()
        self.inpaint_model.eval()
        create_dir(self.results_path)

        inf_loader = DataLoader(
            dataset=self.infer_dataset,
            batch_size=1
        )

        name = os.path.basename(self.config.input_image).replace(".mnc.gz", "-Inpainted.mnc")
        path = os.path.join(self.results_path, name)

        out_image = copy.deepcopy(self.infer_dataset.inf_img)
        # out_count = copy.deepcopy(self.infer_dataset.inf_img) * 0
        #
        # for z in self.infer_dataset.z_slices_with_lesions:
        #     out_image[z :, :] = 0

        index = 0
        for items in inf_loader:

            print("Working on z-slice {0}...".format(self.infer_dataset.z_slices_with_lesions[index]))
            image, edge, mask, orig_image, brain_mask, mean_val, std_val, min_val, max_val, imgh, imgw = self.cuda(*items)

            output = self.inpaint_model(image, edge, mask).detach()
            output_merged = (output * mask) + (image * (1 - mask))
            output_unnorm = (((output_merged * max_val) + min_val) * std_val) + mean_val
            output_final = (output_unnorm * brain_mask) + (orig_image * (1 - brain_mask))
            out_image[self.infer_dataset.z_slices_with_lesions[index], :, :] = pad_and_or_crop(
                    output_final.cpu()[0, 0, self.infer_dataset.half_z_size, :, :], [imgh, imgw]).numpy()
            index += 1

        # out_count = np.where(out_count > 0, out_count, 1)
        # out_image /= out_count
        result_image = sitk.GetImageFromArray(out_image)
        result_image.CopyInformation(self.infer_dataset.sitk_image)

        sitk.WriteImage(result_image, path)

    def test(self):
        for r_id, (lr, exp_decay, dtg_lr, l1_loss_w, fm_loss_w, sty_loss_w, cnt_loss_w, inpaint_adv_loss_w) in enumerate(
                product(*self.param_values)):
            run_id = r_id + 1
            if run_id != self.config.run_id:
                continue

            print("run id:", run_id)
            self.config.LR = lr
            self.config.EXP_DECAY = exp_decay
            self.config.D2G_LR = dtg_lr
            self.config.L1_LOSS_WEIGHT = l1_loss_w
            # self.config.MSE_LOSS_WEIGHT = mse_loss_w
            self.config.FM_LOSS_WEIGHT = fm_loss_w
            self.config.STYLE_LOSS_WEIGHT = sty_loss_w
            self.config.CONTENT_LOSS_WEIGHT = cnt_loss_w
            self.config.INPAINT_ADV_LOSS_WEIGHT = inpaint_adv_loss_w

            self.model_name = "GAN-InPainting-RunID-{0}".format(run_id)
            self.run_path = os.path.join(self.config.PATH, "Run-{0}".format(run_id))
            if self.config.model_3D:
                self.inpaint_model = InpaintingModel3D(
                    self.config, run_path=self.run_path, name=self.model_name).to(self.config.DEVICE)
            else:
                self.inpaint_model = InpaintingModel(
                    self.config, run_path=self.run_path, name=self.model_name).to(self.config.DEVICE)
            self.results_path = os.path.join(self.run_path, 'results')
            self.tensorboard_path = os.path.join(self.config.TENSORBOARD_DIR, 'runs')

            if self.config.RESULTS is not None:
                self.results_path = os.path.join(self.config.RESULTS)

            if self.config.DEBUG is not None and self.config.DEBUG != 0:
                self.debug = True

            self.writer = SummaryWriter(
                comment=f' lr = {lr} l1_loss_w = {l1_loss_w} fm_loss_w = {fm_loss_w} sty_loss_w = {sty_loss_w}'
                        f'cnt_loss_w = {cnt_loss_w} inpaint_adv_loss_w = {inpaint_adv_loss_w}')

            self.load()
            break

        self.inpaint_model.eval()
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index).replace(".mnc.gz", ".tif")
            # in_name = name.replace(".tif", "-in.tif")
            in_masked_name = name.replace(".tif", "-in-masked.tif")
            # orig_out = name.replace(".tif", "-org.tif")
            images, edges, masks = self.cuda(*items)
            index += 1

            inputs = (images * (1 - masks)) + masks
            outputs = self.inpaint_model(images, edges, masks).detach()
            outputs_merged = (outputs * masks) + (images * (1 - masks))

            path = os.path.join(self.results_path, name)
            print(index, name)

            imsave(self.postprocess(outputs_merged)[0], path)
            imsave(self.postprocess(inputs)[0], os.path.join(self.results_path, in_masked_name))
            # imsave(self.postprocess(outputs)[0], os.path.join(self.results_path, orig_out))
            # imsave(self.postprocess(images)[0], os.path.join(self.results_path, in_name))

            images_tb = torch.cat([images, outputs_merged, inputs, outputs], dim=0)
            grid = torchvision.utils.make_grid(images_tb, nrow=4, padding=25)
            self.writer.add_image("images-test-RunID-{0}-{1}".format(run_id, self.config.iteration),
                                  grid, global_step=index)

            if self.debug:
                masked = self.postprocess(images * (1 - masks) + masks)[0]
                fname, fext = name.split('.')
                imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))

        print('\nEnd test....')

    def sample(self, it=None, run_id=0):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.inpaint_model.eval()

        items = next(self.sample_iterator)
        images, edges, masks = self.cuda(*items)

        iteration = self.inpaint_model.iteration
        inputs = (images * (1 - masks)) + masks
        outputs = self.inpaint_model(images, edges, masks).detach()
        outputs_merged = (outputs * masks) + (images * (1 - masks))

        if it is not None:
            iteration = it

        # image_per_row = 2
        # if self.config.SAMPLE_SIZE <= 6:
        #     image_per_row = 1
        #
        # images_pil = stitch_images(
        #     self.postprocess(images),
        #     self.postprocess(outputs_merged),
        #     self.postprocess(inputs),
        #     self.postprocess(edges),
        #     self.postprocess(outputs),
        #     img_per_row = image_per_row
        # )
        #
        # path = os.path.join(self.samples_path, self.model_name)
        # name = os.path.join(path, str(iteration).zfill(5) + ".tif")
        # create_dir(path)
        # print('\nsaving sample ' + name)
        # images_pil.save(name)

        if self.config.model_3D:
            mid_slice = int(self.config.z_size / 2)
            images_tb = torch.cat(
                [images[:, :, mid_slice, :, :], outputs_merged[:, :, mid_slice, :, :], inputs[:, :, mid_slice, :, :],
                 edges[:, :, mid_slice, :, :], outputs[:, :, mid_slice, :, :]], dim=0)
        else:
            images_tb = torch.cat([images, outputs_merged, inputs, edges, outputs], dim=0)
        grid = torchvision.utils.make_grid(images_tb, nrow=self.config.SAMPLE_SIZE, padding=25)
        self.writer.add_image("Images-RunID-{0}".format(run_id), grid, global_step=iteration)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img, is_int=False):
        # [0, 1] => [0, 255]
        if self.config.model_3D:
            img = img.permute(0, 2, 3, 4, 1)
        else:
            img = img.permute(0, 2, 3, 1)

        if is_int:
            return img.int()
        else:
            return img.float()
