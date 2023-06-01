import monai
import torch
import torch.nn as nn
import torchvision.models as models


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, native=False):
        super(StyleLoss, self).__init__()
        self.add_module('classifier', MRIClassifier(native=native))
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_classifier, y_classifier = self.classifier(x), self.classifier(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_classifier['block_2_2']), self.compute_gram(y_classifier['block_2_2']))
        style_loss += self.criterion(self.compute_gram(x_classifier['block_4_4']), self.compute_gram(y_classifier['block_4_4']))
        style_loss += self.criterion(self.compute_gram(x_classifier['block_6_6']), self.compute_gram(y_classifier['block_6_6']))
        style_loss += self.criterion(self.compute_gram(x_classifier['block_8_4']), self.compute_gram(y_classifier['block_8_4']))

        return style_loss


class PerceptualLoss(nn.Module):

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0], native=False):
        super(PerceptualLoss, self).__init__()
        self.add_module('classifier', MRIClassifier(native=native))
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_classifier, y_classifier = self.classifier(x), self.classifier(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_classifier['block_2_1'], y_classifier['block_2_1'])
        content_loss += self.weights[1] * self.criterion(x_classifier['block_4_1'], y_classifier['block_4_1'])
        content_loss += self.weights[2] * self.criterion(x_classifier['block_6_1'], y_classifier['block_6_1'])
        content_loss += self.weights[3] * self.criterion(x_classifier['block_8_1'], y_classifier['block_8_1'])

        return content_loss

class MRIClassifier(torch.nn.Module):

    def __init__(self, native=False):
        super(MRIClassifier, self).__init__()

        if native:
            model = monai.networks.nets.DenseNet121(spatial_dims=2, in_channels=1, out_channels=6).to("cuda")
            checkpoint = torch.load(
                "/scratch/02/paryam/workspace/central-deployment/deep-neural-networks/ImageClassifier/"
                "checkpoints/MRIClassifierNative/MRIClassifierNative_checkpoint_63893.pt")
        else:
            model = monai.networks.nets.DenseNet121(spatial_dims=2, in_channels=1, out_channels=5).to("cuda")
            checkpoint = torch.load("/scratch/02/paryam/workspace/central-deployment/deep-neural-networks"
                                    "/ImageClassifier/checkpoints/MRIClassifier/MRIClassifier_checkpoint_58588.pt")
        model.load_state_dict(checkpoint["state_dict"])
        features = model.features

        self.block_1_1 = torch.nn.Sequential()
        for x in range(4):
            self.block_1_1.add_module(str(x), features[x])

        self.block_2_1 = torch.nn.Sequential()
        for x in range(0, 3):
            self.block_2_1.add_module(str(4 + x), features[4][x])
        self.block_2_2 = torch.nn.Sequential()
        for x in range(3, 6):
            self.block_2_2.add_module(str(4 + x), features[4][x])

        self.block_3_1 = torch.nn.Sequential()
        self.block_3_1.add_module(str(10), features[5])

        self.block_4_1 = torch.nn.Sequential()
        for x in range(0, 3):
            self.block_4_1.add_module(str(11 + x), features[6][x])
        self.block_4_2 = torch.nn.Sequential()
        for x in range(3, 6):
            self.block_4_2.add_module(str(11 + x), features[6][x])
        self.block_4_3 = torch.nn.Sequential()
        for x in range(6, 9):
            self.block_4_3.add_module(str(11 + x), features[6][x])
        self.block_4_4 = torch.nn.Sequential()
        for x in range(9, 12):
            self.block_4_4.add_module(str(11 + x), features[6][x])

        self.block_5_1 = torch.nn.Sequential()
        self.block_5_1.add_module(str(23), features[7])

        self.block_6_1 = torch.nn.Sequential()
        for x in range(0, 4):
            self.block_6_1.add_module(str(24 + x), features[8][x])
        self.block_6_2 = torch.nn.Sequential()
        for x in range(4, 8):
            self.block_6_2.add_module(str(24 + x), features[8][x])
        self.block_6_3 = torch.nn.Sequential()
        for x in range(8, 12):
            self.block_6_3.add_module(str(24 + x), features[8][x])
        self.block_6_4 = torch.nn.Sequential()
        for x in range(12, 16):
            self.block_6_4.add_module(str(24 + x), features[8][x])
        self.block_6_5 = torch.nn.Sequential()
        for x in range(16, 20):
            self.block_6_5.add_module(str(24 + x), features[8][x])
        self.block_6_6 = torch.nn.Sequential()
        for x in range(20, 24):
            self.block_6_6.add_module(str(24 + x), features[8][x])

        self.block_7_1 = torch.nn.Sequential()
        self.block_7_1.add_module(str(48), features[9])

        self.block_8_1 = torch.nn.Sequential()
        for x in range(0, 4):
            self.block_8_1.add_module(str(49 + x), features[10][x])
        self.block_8_2 = torch.nn.Sequential()
        for x in range(4, 8):
            self.block_8_2.add_module(str(49 + x), features[10][x])
        self.block_8_3 = torch.nn.Sequential()
        for x in range(8, 12):
            self.block_8_3.add_module(str(49 + x), features[10][x])
        self.block_8_4 = torch.nn.Sequential()
        for x in range(12, 16):
            self.block_8_4.add_module(str(49 + x), features[10][x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        block_1_1 = self.block_1_1(x)

        block_2_1 = self.block_2_1(block_1_1)
        block_2_2 = self.block_2_2(block_2_1)

        block_3_1 = self.block_3_1(block_2_2)

        block_4_1 = self.block_4_1(block_3_1)
        block_4_2 = self.block_4_2(block_4_1)
        block_4_3 = self.block_4_3(block_4_2)
        block_4_4 = self.block_4_4(block_4_3)

        block_5_1 = self.block_5_1(block_4_4)

        block_6_1 = self.block_6_1(block_5_1)
        block_6_2 = self.block_6_2(block_6_1)
        block_6_3 = self.block_6_3(block_6_2)
        block_6_4 = self.block_6_4(block_6_3)
        block_6_5 = self.block_6_5(block_6_4)
        block_6_6 = self.block_6_6(block_6_5)

        block_7_1 = self.block_7_1(block_6_6)

        block_8_1 = self.block_8_1(block_7_1)
        block_8_2 = self.block_8_2(block_8_1)
        block_8_3 = self.block_8_3(block_8_2)
        block_8_4 = self.block_8_4(block_8_3)

        out = {
            'block_1_1': block_1_1,

            'block_2_1': block_2_1,
            'block_2_2': block_2_2,

            'block_3_1': block_3_1,

            'block_4_1': block_4_1,
            'block_4_2': block_4_2,
            'block_4_3': block_4_3,
            'block_4_4': block_4_4,

            'block_5_1': block_5_1,

            'block_6_1': block_6_1,
            'block_6_2': block_6_2,
            'block_6_3': block_6_3,
            'block_6_4': block_6_4,
            'block_6_5': block_6_6,
            'block_6_6': block_6_6,

            'block_7_1': block_7_1,

            'block_8_1': block_8_1,
            'block_8_2': block_8_2,
            'block_8_3': block_8_3,
            'block_8_4': block_8_4,
        }
        return out
