The lesion inpainting code is partially inspired by:
[https://github.com/knazeri/edge-connect](https://github.com/knazeri/edge-connect)
@InProceedings{Nazeri_2019_ICCV,
    title = {EdgeConnect: Structure Guided Image Inpainting using Edge Prediction},
    author = {Nazeri, Kamyar and Ng, Eric and Joseph, Tony and Qureshi, Faisal and Ebrahimi, Mehran},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV) Workshops},
    month = {Oct},
    year = {2019}
}


# Training code for Lesion Inpainting

This directory contains the code for Lesion inpainting

The inpainting is trained per image modality.
I decided to do the inpainting within slice when spacing between axizl slices is big (t1p, t2w ...)
For t1g, the inpainting was done using the 3D model where a few slices up and down are considered for inpainting.
 
## Architechture

A generative adversarial network has been used to train the model. 
The loss used by the network is the sum of few terms:
1. Adversarial loss
2. l1-loss between fake generated image, and the actual image
3. L1-loss between feature maps of the discriminator network for every fake and real images
4. Perceptual loss 
5. Style loss 
   
The last two loss terms are  computed using a pre-trained ImageClassifier network trained on NeuroRx images 
to classify different image modalities (***Whose training code can be found in the same repository under 
ImageClassifier folder***)

Spectral Normalization has also been to stabilize training of the discriminator.

## Training data
The network has been trained for native t1g, native t1p and stx-registered t2w images and the 
below config files contains the information on training data and other parameters used for training:
1. `/scratch/02/paryam/workspace/central-deployment/deep-neural-networks/GAN-Inpainting/checkpoints/t1g-inpainting-native/config.yml`
2. `/scratch/02/paryam/workspace/central-deployment/deep-neural-networks/GAN-Inpainting/checkpoints/t2w-inpainting/config.yml`
3. `/scratch/02/paryam/workspace/central-deployment/deep-neural-networks/GAN-Inpainting/checkpoints/t1p-inpainting-native/config.yml`

The main challenge was how to define lesion-like areas to use for training.
It was done by copying lesions from other subjects while constraining lesions to fall within the NAWM.
