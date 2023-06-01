import os
import sys
import SimpleITK as sitk
from utils import check_call_out

with open("/scratch/02/paryam/workspace/central-deployment/deep-neural-networks/GAN-Inpainting/Datasets/t1p-inpainting-native/t1p_images_native.txt") as f:
    images = f.readlines()

with open("/scratch/02/paryam/workspace/central-deployment/deep-neural-networks/GAN-Inpainting/Datasets/t1p-inpainting-native/t2_lesions.txt") as f:
    t2_lesions = f.readlines()

print(sys.argv)
if len(sys.argv) > 1 and sys.argv[1] == "-reverse":
    images = images[::-1]
    t2_lesions = t2_lesions[::-1]

minc_dir = "/scratch/02/paryam/conda-envs/miniconda3/envs/pytorch/bin"
for i, file_name in enumerate(images):
    img_file = file_name.rstrip().replace(".mnc.gz", "_ISPC-stx152lsq6.mnc.gz")
    if not os.path.exists(img_file):
        img_file = os.path.join(
            os.path.dirname(img_file), "ImagePrep2/2.0/IntensityNorm/1.0.0/", os.path.basename(img_file))

    minc2_file = img_file.replace(".mnc.gz", "_tmp.mnc")
    nii_file = minc2_file.replace("_tmp.mnc", ".nii")
    if os.path.exists(nii_file):
        print(nii_file + " already exists. \n")
        continue

    if not os.path.exists(minc2_file):
        check_call_out(
            command="{0}/mincconvert -2 {1} {2}".format(minc_dir, img_file, minc2_file))

    print(minc2_file)
    sitk_im = sitk.ReadImage(minc2_file)
    sitk.WriteImage(sitk_im, minc2_file.replace("_tmp.mnc", ".nii"))

    lesion_file = t2_lesions[i].rstrip()
    minc2_file = lesion_file.replace(".mnc.gz", "_ISPC-stx152lsq6_tmp.mnc")

    print(minc2_file)
    sitk_im = sitk.ReadImage(minc2_file)
    sitk.WriteImage(sitk_im, minc2_file.replace("_tmp.mnc", ".nii"))

    print()