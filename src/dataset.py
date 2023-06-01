import bisect
import copy
import json
import os
import tempfile

import skimage.transform
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from imageio import imread
import SimpleITK as sitk
from scipy import ndimage
from skimage.feature import canny
from .utils import check_call_out, pad_and_or_crop


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, img_list, lesion_list, brain_mask_list, tissue_mask_list=None, xfms=None,
                 external_mask_list=None, augment=True, training=True, inference=False):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.inference = inference
        self.images = self.load_list(img_list)
        self.lesions = self.load_list(lesion_list)
        self.brain_masks = self.load_list(brain_mask_list)
        self.tissue_masks = self.load_list(tissue_mask_list)
        self.xfms = self.load_list(xfms)
        self.ext_masks = self.load_list(external_mask_list)

        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.mask = config.MASK
        self.nms = config.NMS
        self.resized = config.RESIZED

        self.img_mod = config.IMG_MODE
        self.z_size = config.z_size
        self.struct = ndimage.generate_binary_structure(2, 1)
        if config.model_3D:
            self.half_z_size = int(config.z_size / 2)
            self.struct = ndimage.generate_binary_structure(3, 2)

        self.minc_dir = config.MINC_DIR
        self.min_mask_size = config.MIN_MASK_SIZE

        self.ct2f_slices = dict()
        with open(config.LESION_SLICES, 'r') as fp:
            self.ct2f_slices = json.load(fp)

        self.lesional_slices = dict()
        with open(config.LESIONAL_SLICES_PER_IMAGE, 'r') as fp:
            self.lesional_slices = json.load(fp)

        if config.MODE == 2:
            self.mask = 1

        self.ind = 0

        self.native = config.NATIVE
        if inference:
            if config.NATIVE:
                self.set_inference_data_native()
            else:
                self.set_inference_data()

    def set_inference_data_native(self):

        mr_image = self.images[0]
        ct2f_label = self.lesions[0]
        brain_mask = self.brain_masks[0]
        tissue_mask = self.tissue_masks[0]
        xfm = self.xfms[0]

        tmp_mr = mr_image.replace(".mnc.gz", "_tmp.mnc")
        if not os.path.exists(tmp_mr):
            print(mr_image)
            check_call_out(command="{0}/mincconvert -2 {1} {2}".format(self.minc_dir, mr_image, tmp_mr))

        if "_ISPC-stx152lsq6" in ct2f_label:
            tmp_ct2f = ct2f_label.replace(".mnc.gz", "_tmp.mnc")
        else:
            tmp_ct2f = ct2f_label.replace(".mnc.gz", "_ISPC-stx152lsq6_tmp.mnc")

        if not os.path.exists(tmp_ct2f):
            print(ct2f_label)
            check_call_out(command="{0}/mincresample -clobber -nearest -like {1} {2} /tmp/{3}".format(
                self.minc_dir, mr_image, ct2f_label, os.path.basename(tmp_ct2f)))
            check_call_out(command="{0}/mincconvert -2 /home/paryam/{1} {2}".format(
                self.minc_dir, os.path.basename(tmp_ct2f), tmp_ct2f))
            os.remove("/tmp/{0}".format(os.path.basename(tmp_ct2f)))

        t2_lesion_resampled = tmp_ct2f.replace("ISPC-stx152lsq6", "ISPC-{0}".format(self.img_mod))
        if not os.path.exists(t2_lesion_resampled):
            check_call_out(
                command="{0}/mincresample -nearest -invert -like {1} {2} {3} -transform {4}".format(
                    self.minc_dir, tmp_mr, tmp_ct2f, t2_lesion_resampled, xfm))

        tmp_bm = brain_mask.replace(".mnc.gz", "_tmp.mnc")
        if not os.path.exists(tmp_bm):
            print(brain_mask)
            check_call_out(command="{0}/mincconvert -2 {1} {2}".format(self.minc_dir, brain_mask, tmp_bm))
        self.cur_bm = tmp_bm

        bm_native = tmp_bm.replace("ISPC-stx152lsq6", "ISPC-{0}".format(self.img_mod))
        if not os.path.exists(bm_native):
            check_call_out(command="{0}/mincresample -invert -nearest -like {1} {2} {3} -transform {4}".format(
                self.minc_dir, tmp_mr, tmp_bm, bm_native, xfm))

        tissue_native = tissue_mask.replace("ISPC-stx152lsq6", "ISPC-{0}".format(self.img_mod))
        if not os.path.exists(tissue_native):
            check_call_out(command="{0}/mincresample -invert -nearest -like {1} {2} {3} -transform {4}".format(
                self.minc_dir, tmp_mr, tissue_mask, tissue_native, xfm))

        self.sitk_image = sitk.ReadImage(tmp_mr)
        self.inf_img = sitk.GetArrayFromImage(self.sitk_image)
        self.inf_t2_lesion = sitk.GetArrayFromImage(sitk.ReadImage(t2_lesion_resampled))
        self.inf_brain_mask = sitk.GetArrayFromImage(sitk.ReadImage(bm_native))

        tissue_mask_native = sitk.GetArrayFromImage(sitk.ReadImage(tissue_native))
        self.inf_tissue_mask = np.where(np.logical_or(
            tissue_mask_native == 3, tissue_mask_native == 6), 1, 0) + self.inf_t2_lesion
        self.inf_tissue_mask = np.where(self.inf_tissue_mask > 0, 1, 0)
        self.z_slices_with_lesions = np.unique(np.where(self.inf_t2_lesion > 0.5)[0])

    def set_inference_data(self):

        mr_image = self.images[0]
        ct2f_label = self.lesions[0]
        brain_mask = self.brain_masks[0]
        tissue_mask = self.tissue_masks[0]

        tmp_mr = mr_image.replace(".mnc.gz", "_tmp.mnc")
        if not os.path.exists(tmp_mr):
            print(mr_image)
            check_call_out(command="{0}/mincconvert -2 {1} {2}".format(self.minc_dir, mr_image, tmp_mr))

        if "_ISPC-stx152lsq6" in ct2f_label:
            tmp_ct2f = ct2f_label.replace(".mnc.gz", "_tmp.mnc")
        else:
            tmp_ct2f = ct2f_label.replace(".mnc.gz", "_ISPC-stx152lsq6_tmp.mnc")
        self.cur_ct2f = tmp_ct2f
        if not os.path.exists(tmp_ct2f):
            print(ct2f_label)
            check_call_out(command="{0}/mincresample -clobber -nearest -like {1} {2} /home/paryam/{3}".format(
                self.minc_dir, mr_image, ct2f_label, os.path.basename(tmp_ct2f)))
            check_call_out(command="{0}/mincconvert -2 /home/paryam/{1} {2}".format(
                self.minc_dir, os.path.basename(tmp_ct2f), tmp_ct2f))
            os.remove("/home/paryam/{0}".format(os.path.basename(tmp_ct2f)))

        tmp_bm = brain_mask.replace(".mnc.gz", "_tmp.mnc")
        if not os.path.exists(tmp_bm):
            print(brain_mask)
            check_call_out(command="{0}/mincconvert -2 {1} {2}".format(self.minc_dir, brain_mask, tmp_bm))

        tmp_tissue = tissue_mask.replace(".mnc.gz", "_tmp.mnc")
        if not os.path.exists(tmp_tissue):
            print(tissue_mask)
            check_call_out(command="{0}/mincconvert -2 {1} {2}".format(self.minc_dir, tissue_mask, tmp_tissue))

        self.sitk_image = sitk.ReadImage(tmp_mr)
        self.inf_img = sitk.GetArrayFromImage(self.sitk_image)
        self.inf_t2_lesion = sitk.GetArrayFromImage(sitk.ReadImage(tmp_ct2f))
        self.inf_brain_mask = sitk.GetArrayFromImage(sitk.ReadImage(tmp_bm))

        tissue_mask = sitk.GetArrayFromImage(sitk.ReadImage(tmp_tissue))
        self.inf_tissue_mask = np.where(np.logical_or(
            tissue_mask == 3, tissue_mask == 6), 1, 0) + self.inf_t2_lesion
        self.inf_tissue_mask = np.where(self.inf_tissue_mask > 0, 1, 0)
        self.z_slices_with_lesions = np.unique(np.where(self.inf_t2_lesion > 0.5)[0])

    def __len__(self):
        if self.inference:
            return len(self.z_slices_with_lesions)
        else:
            return len(self.images)

    def __getitem__(self, index):
        if self.inference:
            if self.native:
                if self.z_size == 1:
                    item = self.load_inf_item_native(index)
                else:
                    item = self.load_inf_item_native_3D(index)
            else:
                item = self.load_inf_item(index)
        else:
            item = self.load_item(index)
        return item

    def load_name(self, index):
        name = self.images[index]
        return os.path.basename(name)

    def _normalize(self, image, mask, return_norm_values=False):
        masked_img = np.ma.masked_where(mask==0, image)
        mean_value = np.ma.mean(masked_img)
        std_value = np.ma.std(masked_img)
        if np.ma.std(masked_img) > 0:
            masked_img = (masked_img - np.ma.mean(masked_img)) / (np.ma.std(masked_img))

        min_value = np.ma.min(masked_img)
        masked_img -= np.ma.min(masked_img)

        max_value = np.ma.max(masked_img)
        if np.ma.max(masked_img) > 0:
            masked_img /= np.ma.max(masked_img)
        else:
            masked_img += 0.0001
        image = masked_img.filled(0)
        if return_norm_values:
            return image, mean_value, std_value, min_value, max_value
        else:
            return image

    def load_inf_item_native_3D(self, index):
        imgh, imgw = self.inf_img.shape[1:3]
        chosen_z_slice = self.z_slices_with_lesions[index]
        masked_region = self.inf_t2_lesion[chosen_z_slice - self.half_z_size: chosen_z_slice + self.half_z_size, :, :]
        masked_region = ndimage.binary_dilation(
            masked_region, structure=self.struct, iterations=1) * \
                        self.inf_tissue_mask[
                        chosen_z_slice - self.half_z_size: chosen_z_slice + self.half_z_size, :, :].astype(np.uint8)
        chosen_img_slices = (
                self.inf_img[chosen_z_slice - self.half_z_size: chosen_z_slice + self.half_z_size, :, :] *
                self.inf_brain_mask[chosen_z_slice - self.half_z_size: chosen_z_slice + self.half_z_size, :, :]).reshape(
            self.z_size, imgh, imgw)

        chosen_img_slices = pad_and_or_crop(chosen_img_slices, [self.z_size] + self.resized)
        brain_slices = pad_and_or_crop(
            self.inf_brain_mask[chosen_z_slice - self.half_z_size: chosen_z_slice + self.half_z_size, :, :],
            [self.z_size] + self.resized)
        masked_region = pad_and_or_crop(masked_region, [self.z_size] + self.resized)
        orig_img = pad_and_or_crop(
            self.inf_img[chosen_z_slice - self.half_z_size: chosen_z_slice + self.half_z_size, :, :],
            [self.z_size] + self.resized)
        chosen_img_slices, mean_value, std_value, min_value, max_value = self._normalize(
            chosen_img_slices, brain_slices, return_norm_values=True)
        edge = self.load_edge(chosen_img_slices, masked_region)

        return self.to_tensor(chosen_img_slices.astype(np.float32)), self.to_tensor(edge), \
               self.to_tensor(masked_region.astype(np.float32)), self.to_tensor(orig_img), \
               self.to_tensor(brain_slices), mean_value, std_value, min_value, max_value, imgh, imgw

    def load_inf_item_native(self, index):
        imgh, imgw = self.inf_img.shape[1:3]
        chosen_z_slice = self.z_slices_with_lesions[index]
        masked_region = self.inf_t2_lesion[chosen_z_slice, :, :]
        masked_region = ndimage.binary_dilation(
            masked_region, structure=self.struct, iterations=1) \
                        * self.inf_tissue_mask[chosen_z_slice, :, :].astype(np.uint8)
        chosen_img_slice = (
                self.inf_img[chosen_z_slice, :, :] * self.inf_brain_mask[chosen_z_slice, :, :]).reshape(imgh, imgw)

        chosen_img_slice = pad_and_or_crop(chosen_img_slice, self.resized)
        brain_slice = pad_and_or_crop(self.inf_brain_mask[chosen_z_slice, :, :], self.resized)
        masked_region = pad_and_or_crop(masked_region, self.resized)
        orig_img = pad_and_or_crop(self.inf_img[chosen_z_slice, :, :], self.resized)
        chosen_img_slice, mean_value, std_value, min_value, max_value = self._normalize(
            chosen_img_slice, brain_slice, return_norm_values=True)
        edge = self.load_edge(chosen_img_slice, masked_region)

        return self.to_tensor(chosen_img_slice.astype(np.float32)), self.to_tensor(edge), \
               self.to_tensor(masked_region.astype(np.float32)), self.to_tensor(orig_img), \
               self.to_tensor(brain_slice), mean_value, std_value, min_value, max_value, imgh, imgw

    def load_inf_item(self, index):
        imgh, imgw = self.inf_img.shape[1:3]
        chosen_z_slice = self.z_slices_with_lesions[index]
        masked_region = self.inf_t2_lesion[chosen_z_slice, :, :]
        masked_region = ndimage.binary_dilation(masked_region, structure=self.struct, iterations=1) \
                        * self.inf_tissue_mask[chosen_z_slice, :, :].astype(np.uint8)
        chosen_img_slice = (
                self.inf_img[chosen_z_slice, :, :] * self.inf_brain_mask[chosen_z_slice, :, :]).reshape(imgh, imgw)
        chosen_img_slice, mean_value, std_value, min_value, max_value= self._normalize(
            chosen_img_slice, self.inf_brain_mask[chosen_z_slice, :, :], return_norm_values=True)
        edge = self.load_edge(chosen_img_slice, masked_region)

        return self.to_tensor(chosen_img_slice.astype(np.float32)), self.to_tensor(edge), \
               self.to_tensor(masked_region.astype(np.float32)), self.to_tensor(self.inf_img[chosen_z_slice, :, :]), \
               self.to_tensor(self.inf_brain_mask[chosen_z_slice, :, :]), \
               mean_value, std_value, min_value, max_value, imgh, imgw

    def load_item(self, index):
        self.ind = index

        mr_image = self.images[index]
        ct2f_label = self.lesions[index]
        brain_mask = self.brain_masks[index]
        xfm = None
        tissue_mask = None
        if self.xfms:
            xfm = self.xfms[index]
        if self.tissue_masks:
            tissue_mask = self.tissue_masks[index]

        tmp_mr = mr_image.replace(".mnc.gz", "_tmp.mnc")
        if not os.path.exists(tmp_mr):
            print(mr_image)
            check_call_out(command="{0}/mincconvert -2 {1} {2}".format(self.minc_dir, mr_image, tmp_mr))

        tmp_ct2f = ct2f_label.replace(".mnc.gz", "_ISPC-stx152lsq6_tmp.mnc")
        self.cur_ct2f = tmp_ct2f
        self.cur_image = tmp_mr

        if not os.path.exists(tmp_ct2f):
            print(ct2f_label)
            check_call_out(command="{0}/mincresample -clobber -nearest -like {1} {2} /home/paryam/{3}".format(
                self.minc_dir, mr_image, ct2f_label, os.path.basename(tmp_ct2f)))
            check_call_out(command="{0}/mincconvert -2 /home/paryam/{1} {2}".format(
                self.minc_dir, os.path.basename(tmp_ct2f), tmp_ct2f))
            os.remove("/home/paryam/{0}".format(os.path.basename(tmp_ct2f)))

        tmp_bm = brain_mask.replace(".mnc.gz", "_tmp.mnc")
        if not os.path.exists(tmp_bm):
            print(brain_mask)
            check_call_out(command="{0}/mincconvert -2 {1} {2}".format(self.minc_dir, brain_mask, tmp_bm))
        self.cur_bm = tmp_bm

        brain_mask_native = None
        if self.xfms:
            bm_native = tmp_bm.replace("ISPC-stx152lsq6", "ISPC-{0}".format(self.img_mod))
            if not os.path.exists(bm_native):
                check_call_out(command="{0}/mincresample -invert -nearest -like {1} {2} {3} -transform {4}".format(
                    self.minc_dir, self.cur_image, tmp_bm, bm_native, xfm))
            brain_mask_native = sitk.GetArrayFromImage(sitk.ReadImage(bm_native))

        img = sitk.GetArrayFromImage(sitk.ReadImage(tmp_mr))
        t2_lesion = sitk.GetArrayFromImage(sitk.ReadImage(tmp_ct2f))
        brain_mask = sitk.GetArrayFromImage(sitk.ReadImage(tmp_bm))
        if tissue_mask:
            tissue_mask = sitk.GetArrayFromImage(sitk.ReadImage(tissue_mask))

        mask, img = self.load_image_and_mask(img, t2_lesion, brain_mask,
                                             brain_mask_native=brain_mask_native, xfm=xfm, tissue_mask=tissue_mask)
        edge = self.load_edge(img, mask)

        # if self.augment and np.random.binomial(1, 0.5) > 0:
        #     img = img[:, ::-1, ...]
        #     edge = edge[:, ::-1, ...]
        #     mask = mask[:, ::-1, ...]

        return self.to_tensor(img.astype(np.float32)), \
               self.to_tensor(edge), self.to_tensor(mask.astype(np.float32))

    def load_edge(self, img, mask):
        sigma = self.sigma

        if sigma == -1:
            return np.zeros(img.shape).astype(np.float)

        # random sigma
        if sigma == 0:
            sigma = random.randint(1, 4)

        mask = ndimage.binary_dilation(mask, structure=self.struct, iterations=2).astype(mask.dtype)
        if self.z_size == 1:
            mask = (1 - mask).astype(np.bool)
            return canny(img, sigma=sigma, mask=mask).astype(np.float32)
        else:
            canny_filter = sitk.CannyEdgeDetectionImageFilter()
            canny_filter.SetVariance(sigma)
            canny_edges = sitk.GetArrayFromImage(canny_filter.Execute(sitk.GetImageFromArray(img)))
            masked_edges = np.where(mask > 0, 0, canny_edges)
            return masked_edges

    def load_image_and_mask(self, img, t2_lesion, brain_mask, xfm=None, brain_mask_native=None, tissue_mask=None):
        imgh, imgw = img.shape[1:3]

        mask_type = 1
        if xfm:
            found, chosen_z_slice, mask, image = self.load_slice_and_mask_from_another_in_native(
                img, t2_lesion, brain_mask, brain_mask_native, xfm, tissue_mask)

            if self.z_size == 1:
                image = pad_and_or_crop(image, self.resized)
                mask = pad_and_or_crop(mask, self.resized)
                brain_mask_native = pad_and_or_crop(brain_mask_native[chosen_z_slice, :, :], self.resized)
            else:
                image = pad_and_or_crop(image, [self.z_size] + self.resized)
                mask = pad_and_or_crop(mask, [self.z_size] + self.resized)
                brain_mask_native = pad_and_or_crop(
                    brain_mask_native[chosen_z_slice - self.half_z_size: chosen_z_slice + self.half_z_size, :, :],
                    [self.z_size] + self.resized)

            image = self._normalize(image, brain_mask_native)
        else:
            found, chosen_z_slice, mask, image = self.load_slice_and_mask_from_another(
                img, t2_lesion, brain_mask, tissue_mask)

        if mask_type == 1:
            if found or not self.training:
                return mask.astype(np.uint8), image
            else:
                mask_type = 3

        # external
        if mask_type == 2:
            mask_index = random.randint(0, len(self.ext_masks) - 1)
            ext_mask = imread(self.ext_masks[mask_index])
            ext_mask = self.resize(ext_mask, imgh, imgw)
            ext_mask = 1 - (ext_mask > 0).astype(np.uint8)
            mask = ext_mask + mask if mask is not None else ext_mask
            bm_slice = brain_mask[chosen_z_slice, :, :]
            mask = np.where(bm_slice == 0, 0, mask).astype(np.uint8)
            if np.sum(mask) > self.min_mask_size:
                return mask, image
            mask_type = 3

        if mask_type == 3:
            if xfm:
                rand_mask = self.create_random_mask(brain_mask_native)
            else:
                rand_mask = self.create_random_mask(
                    brain_mask[chosen_z_slice - self.half_z_size: chosen_z_slice + self.half_z_size, :, :])
            mask = rand_mask + mask if mask is not None else rand_mask
            mask = (mask > 0).astype(np.uint8)
            return mask, image

    def add_lesions_from_other_images(self, t2_lesion, brain_mask, xfm, tissue_mask=None, clobber=False):

        tmp_lesion = self.cur_ct2f.split("_tmp.mnc")[0] + "_simulated.mnc"
        tmp_lesion_resampled = tmp_lesion.replace("ISPC-stx152lsq6", "ISPC-{0}".format(self.img_mod))
        if os.path.exists(tmp_lesion_resampled):
            if not clobber:
                t2_lesion_simulated = sitk.GetArrayFromImage(sitk.ReadImage(tmp_lesion_resampled))
                z_slices_with_lesions = np.unique(np.where(t2_lesion_simulated > 0.5)[0])
                return  t2_lesion_simulated, z_slices_with_lesions
            else:
                os.remove(tmp_lesion)
                os.remove(tmp_lesion_resampled)

        other_ct2fs = random.sample(self.lesional_slices.keys(), k=2)
        other_ct2fs = [other for other in other_ct2fs if other != self.cur_ct2f]
        if not other_ct2fs:
            other_ct2fs = random.sample(self.lesional_slices.keys(), k=2)
            other_ct2fs = [other for other in other_ct2fs if other != self.cur_ct2f]

        other_t2_lesions = []
        for other_ct2f in other_ct2fs:
            other_t2_lesions.append(sitk.GetArrayFromImage(sitk.ReadImage(other_ct2f)))

        total_lesion = np.zeros(t2_lesion.shape)
        for i, other_t2_lesion in enumerate(other_t2_lesions):
            total_lesion = total_lesion + other_t2_lesion

        total_lesion = np.where(total_lesion > 0, 1, 0)
        if tissue_mask is not None:
            t2_lesion_simulated = (
                    np.where(t2_lesion > 0.5, 0, total_lesion) *
                    np.where(np.logical_or(tissue_mask==3, tissue_mask==6), 1, 0)).astype(tissue_mask.dtype)
        else:
            t2_lesion_simulated = (np.where(t2_lesion > 0.5, 0, total_lesion) * brain_mask).astype(brain_mask.dtype)

        t2lesion_simulated_sitk = sitk.GetImageFromArray(t2_lesion_simulated)
        t2lesion_simulated_sitk.CopyInformation(sitk.ReadImage(self.cur_bm))

        sitk.WriteImage(t2lesion_simulated_sitk, tmp_lesion)

        check_call_out(
            command="{0}/mincresample -nearest -invert -like {1} {2} {3} -transform {4}".format(
                self.minc_dir, self.cur_image, tmp_lesion, tmp_lesion_resampled, xfm))

        t2_lesion_simulated = sitk.GetArrayFromImage(sitk.ReadImage(tmp_lesion_resampled))
        z_slices_with_lesions = np.unique(np.where(t2_lesion_simulated > 0.5)[0])

        return t2_lesion_simulated, z_slices_with_lesions

    def _use_load_slice_and_mask_from_another(self, xfm, brain_mask_native, img):
        t2_lesion_resampled = self.cur_ct2f.replace("ISPC-stx152lsq6", "ISPC-{0}".format(self.img_mod))
        if not os.path.exists(t2_lesion_resampled):
            check_call_out(
                command="{0}/mincresample -nearest -invert -like {1} {2} {3} -transform {4}".format(
                    self.minc_dir, self.cur_image, self.cur_ct2f, t2_lesion_resampled, xfm))

        t2_lesion_res = sitk.GetArrayFromImage(sitk.ReadImage(t2_lesion_resampled))
        found, chosen_z_slice, masked_region, chosen_img = self.load_slice_and_mask_from_another(
            img, t2_lesion_res, brain_mask_native, normalize=False)
        return found, chosen_z_slice, masked_region, chosen_img

    def load_slice_and_mask_from_another_in_native(
            self, img, t2_lesion, brain_mask, brain_mask_native, xfm, tissue_mask=None):

        if not self.training:
            found, chosen_z_slice, masked_region, chosen_img = self._use_load_slice_and_mask_from_another(
                xfm, brain_mask_native, img)
            return found, chosen_z_slice, masked_region, chosen_img

        imgh, imgw = img.shape[1:3]
        found = False
        t2_lesion_simulated, z_slices_with_lesions = self.add_lesions_from_other_images(
            t2_lesion, brain_mask, xfm, tissue_mask=tissue_mask)

        if len(z_slices_with_lesions) == 0:
            t2_lesion_simulated, z_slices_with_lesions = self.add_lesions_from_other_images(
                t2_lesion, brain_mask, xfm, tissue_mask=tissue_mask, clobber=False)

        weights = [10 * pow(1 - np.abs(i/len(z_slices_with_lesions) - 0.5), 3)
                   for i in range(len(z_slices_with_lesions))]
        try:
            chosen_z_slice = random.choices(z_slices_with_lesions, weights=weights, k=1)[0]
        except IndexError:
            found, chosen_z_slice, masked_region, chosen_img = self._use_load_slice_and_mask_from_another(
                xfm, brain_mask_native, img)
            return found, chosen_z_slice, masked_region, chosen_img

        if self.z_size == 1:
            masked_region = t2_lesion_simulated[chosen_z_slice, :, :]
            bm_slice = brain_mask_native[chosen_z_slice, :, :]
            masked_region = np.where(bm_slice == 0, 0, masked_region)
        else:
            masked_region = t2_lesion_simulated[
                            chosen_z_slice - self.half_z_size: chosen_z_slice + self.half_z_size, :, :]
            bm_slice = brain_mask_native[
                       chosen_z_slice - self.half_z_size: chosen_z_slice + self.half_z_size, :, :]
            masked_region = np.where(bm_slice == 0, 0, masked_region)

        if np.random.binomial(1, 0.5) > 0 or np.sum(masked_region) == 0:
            if np.sum(masked_region) > self.min_mask_size:
                found = True
            else:
                masked_region = ndimage.binary_dilation(masked_region, structure=self.struct, iterations=3)
                masked_region = np.where(bm_slice == 0, 0, masked_region)
                if np.sum(masked_region) > self.min_mask_size:
                    found = True
        else:
            found = True

        if self.z_size == 1:
            chosen_img = (img[chosen_z_slice, :, :] * brain_mask_native[chosen_z_slice, :, :]).reshape(imgh, imgw)
        else:
            chosen_img = (
                    img[chosen_z_slice - self.half_z_size: chosen_z_slice + self.half_z_size, :, :] *
                    brain_mask_native[chosen_z_slice - self.half_z_size: chosen_z_slice + self.half_z_size, :, :]).reshape(
            self.z_size, imgh, imgw)
        return found, chosen_z_slice, masked_region, chosen_img

    def load_slice_and_mask_from_another(self, img, t2_lesion, brain_mask, tissue_mask=None, normalize=True):

        imgh, imgw = img.shape[1:3]
        z_slices_with_lesions = np.unique(np.where(t2_lesion > 0.5)[0])
        if tissue_mask is not None:
            z_slices_bm = np.unique(np.where(
                np.logical_or(tissue_mask * brain_mask == 3, tissue_mask * brain_mask == 6))[0])
        else:
            z_slices_bm = np.unique(np.where(brain_mask > 0)[0])
        z_slices_without_lesion = list(set(z_slices_bm) - set(z_slices_with_lesions))

        found = False
        if len(z_slices_with_lesions) and not self.training:
            chosen_z_slice = random.choice(z_slices_with_lesions)
            if self.z_size == 1:
                masked_region = t2_lesion[chosen_z_slice, :, :]
            else:
                masked_region = t2_lesion[
                                chosen_z_slice - self.half_z_Size: chosen_z_slice + self.half_z_size, :, :]
            found = True
        else:
            if not self.training:
                print("Test Image has no lesion: {}".format(self.images[self.ind]))

            weights = [10 * (1 - np.abs(i/len(z_slices_without_lesion) - 0.5)) * (1 - np.abs(i/len(z_slices_without_lesion) - 0.5))
                       for i in range(len(z_slices_without_lesion))]
            chosen_z_slice = random.choices(
                z_slices_without_lesion, weights=weights, k=1)[0]

            try:
                other_ct2fs = random.sample(self.ct2f_slices["{0}".format(chosen_z_slice)], k = 2)
                other_ct2f = [other for other in other_ct2fs if other != self.cur_ct2f][0]
                other_t2_lesion = sitk.GetArrayFromImage(sitk.ReadImage(other_ct2f))

                masked_region = other_t2_lesion[chosen_z_slice, :, :]
                if tissue_mask is not None:
                    tissue_slice = tissue_mask[chosen_z_slice, :, :]
                    masked_region = np.where(np.logical_or(tissue_slice != 3, tissue_slice != 6), 0, masked_region)
                else:
                    bm_slice = brain_mask[chosen_z_slice, :, :]
                    masked_region = np.where(bm_slice == 0, 0, masked_region)

                if np.random.binomial(1, 0.5) > 0 or np.sum(masked_region) == 0:
                    if np.sum(masked_region) > self.min_mask_size:
                        found = True
                    else:
                        masked_region = ndimage.binary_dilation(masked_region, structure=self.struct, iterations=3)
                        if tissue_mask is not None:
                            masked_region = np.where(
                                np.logical_or(tissue_slice != 3, tissue_slice != 6), 0, masked_region)
                        else:
                            masked_region = np.where(bm_slice == 0, 0, masked_region)
                        if np.sum(masked_region) > self.min_mask_size:
                            found = True
                else:
                    found = True
            except (ValueError, KeyError):
                masked_region = np.zeros(shape=(imgh, imgw), dtype=np.int32)

        if self.z_size == 1:
            chosen_img = (img[chosen_z_slice, :, :] * brain_mask[chosen_z_slice, :, :]).reshape(imgh, imgw)
            if normalize:
                chosen_img = self._normalize(chosen_img, brain_mask[chosen_z_slice, :, :])
        else:
            chosen_img = (
                    img[chosen_z_slice - self.half_z_Size: chosen_z_slice + self.half_z_size, :, :] *
                    brain_mask[chosen_z_slice - self.half_z_Size: chosen_z_slice + self.half_z_size, :, :]).reshape(
                self.z_size, imgh, imgw)
            if normalize:
                chosen_img = self._normalize(chosen_img, brain_mask[chosen_z_slice, :, :])

        return found, chosen_z_slice, masked_region, chosen_img

    def load_slice_and_mask(self, img, t2_lesion, brain_mask):

        imgh, imgw = img.shape[1:3]
        z_slices_with_lesions = np.unique(np.where(t2_lesion > 0.5)[0])
        z_slices_bm = np.unique(np.where(brain_mask > 0)[0])
        z_slices_without_lesion = list(set(z_slices_bm) - set(z_slices_with_lesions))

        found = False
        masked_region = np.zeros(shape=(imgh, imgw), dtype=np.int32)
        if len(z_slices_with_lesions):
            if not self.training:
                chosen_z_slice = random.choice(z_slices_with_lesions)
                masked_region = t2_lesion[chosen_z_slice, :, :]
                found = True
            else:
                ct2f_slice = random.choice(z_slices_with_lesions)
                masked_region = t2_lesion[ct2f_slice, :, :]

                location = bisect.bisect_left(z_slices_without_lesion, ct2f_slice)
                l_ind = min(location, len(z_slices_without_lesion)-1)
                r_ind = max(0, location-1)
                if (ct2f_slice - z_slices_without_lesion[r_ind]) < (z_slices_without_lesion[l_ind] - ct2f_slice):
                    chosen_z_slice = z_slices_without_lesion[r_ind]
                else:
                    chosen_z_slice = z_slices_without_lesion[l_ind]

                bm_slice = brain_mask[chosen_z_slice, :, :]
                masked_region = np.where(bm_slice == 0, 0, masked_region)

                if np.sum(masked_region) > self.min_mask_size:
                    found = True
                else:
                    masked_region = ndimage.binary_dilation(masked_region, structure=self.struct, iterations=3)
                    masked_region = np.where(bm_slice == 0, 0, masked_region)
                    if np.sum(masked_region) > self.min_mask_size:
                        found = True

                masked_region = masked_region.reshape(imgh, imgw)
        else:
            weights = [1 - np.abs(i/len(z_slices_without_lesion) - 0.5)
                     for i in range(len(z_slices_without_lesion))]
            chosen_z_slice = random.choices(
                z_slices_without_lesion, weights=weights, k=1)[0]

        chosen_img = (img[chosen_z_slice, :, :] * brain_mask[chosen_z_slice, :, :]).reshape(imgh, imgw)
        return found, chosen_z_slice, masked_region, chosen_img

    def create_random_mask(self, brain_mask):

        bm_slice_eroded = ndimage.binary_erosion(brain_mask, structure=self.struct, iterations=10)
        if not np.sum(bm_slice_eroded):
            bm_slice_eroded = brain_mask
        ind_choices = np.where(bm_slice_eroded>0)
        ind_choices = list(np.transpose(np.asarray(ind_choices)))
        masked_region = np.zeros(brain_mask.shape)
        try:
            chosen_indices = random.sample(ind_choices, 2)
            masked_region[chosen_indices[0][0], chosen_indices[0][1]] = 1
            masked_region[chosen_indices[1][0], chosen_indices[1][1]] = 1
        except ValueError:
            try:
                chosen_indices = random.sample(ind_choices, 1)
                masked_region[chosen_indices[0][0], chosen_indices[0][1]] = 1
            except ValueError:
                print("Error in CreateRandomMask: {0}".format(self.cur_image))

        masked_region = ndimage.binary_dilation(masked_region, structure=self.struct, iterations=4)

        masked_region = np.where(brain_mask == 0, 0, masked_region)

        return masked_region

    def to_tensor(self, img):
        if self.z_size == 1:
            img_t = torch.tensor(img.reshape(1, self.input_size, self.input_size), dtype=torch.float32)
        else:
            img_t = torch.tensor(img.reshape(1, self.z_size, self.input_size, self.input_size), dtype=torch.float32)
        return img_t

    def resize(self, img, height, width):
        imgh, imgw = img.shape[0:2]

        if imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = skimage.transform.resize(img, [height, width], order=0)
        return img

    @staticmethod
    def load_list(file_list):
        if isinstance(file_list, list):
            return file_list

        if isinstance(file_list, str):
            with open(file_list, "r") as f:
                lines = f.readlines()

            list_of_files = [l.rstrip() for l in lines]
            return list_of_files

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
