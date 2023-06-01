import numpy as np
import os
import SimpleITK as sitk
import json

from utils import check_call_out

minc_dir = "/scratch/02/paryam/conda-envs/miniconda3/envs/pytorch/bin/"
with open("/scratch/02/paryam/workspace/central-deployment/deep-neural-networks/"
          "edge-connect/datasets/lesions_final.txt", "r") as f:
    all_lesions = f.readlines()

def summarize_lesions_in_every_stx_slice(all_lesions, minc_dir):
    ref_ispc = '/scratch/02/paryam/datasets/t2net/trials/101MS205/176-SSN-1/' \
               'plb_176-003/baseline/101MS205_176-SSN-1_plb_176-003_baseline_t2w_ISPC-stx152lsq6.mnc.gz'

    all_lesions = [lesion.rstrip() for lesion in all_lesions]
    lesion_slice_dict = dict()
    for lesion_file in all_lesions:
        tmp_lesion_file = lesion_file.replace(".mnc.gz", "_ISPC-stx152lsq6_tmp.mnc")
        if not os.path.exists(tmp_lesion_file):
            print(lesion_file)
            check_call_out(command="{0}/mincresample -clobber -nearest -like {1} {2} /home/paryam/{3}".format(
                minc_dir, ref_ispc, lesion_file, os.path.basename(tmp_lesion_file)))
            check_call_out(command="{0}/mincconvert -2 /home/paryam/{1} {2}".format(
                minc_dir, os.path.basename(tmp_lesion_file), tmp_lesion_file))
            os.remove("/home/paryam/{0}".format(os.path.basename(tmp_lesion_file)))

        t2_lesion = sitk.GetArrayFromImage(sitk.ReadImage(tmp_lesion_file))
        z_slices_with_lesions = np.unique(np.where(t2_lesion > 0.5)[0])

        for z_slice in z_slices_with_lesions:
            lesion_slice_dict.setdefault("{0}".format(z_slice), []).append(tmp_lesion_file)

    with open("/scratch/02/paryam/workspace/central-deployment/deep-neural-networks/"
              "edge-connect/datasets/lesions_slices.json", 'w') as fp:
        fp.write(json.dumps(lesion_slice_dict))

def lesional_slices_per_image(all_lesions):

    all_lesions = [lesion.rstrip() for lesion in all_lesions]

    lesional_slices_dict = dict()
    for lesion_file in all_lesions:
        tmp_lesion_file = lesion_file.replace(".mnc.gz", "_ISPC-stx152lsq6_tmp.mnc")
        t2_lesion = sitk.GetArrayFromImage(sitk.ReadImage(tmp_lesion_file))
        z_slices_with_lesions = list(np.unique(np.where(t2_lesion > 0.5)[0]))

        if z_slices_with_lesions:
            lesional_slices_dict[tmp_lesion_file] = ["{0}".format(s) for s in z_slices_with_lesions]

    with open("/scratch/02/paryam/workspace/central-deployment/deep-neural-networks/"
              "edge-connect/datasets/lesional_slices_per_image.json", 'w') as fp:
        fp.write(json.dumps(lesional_slices_dict))

lesional_slices_per_image(all_lesions)
