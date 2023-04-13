# import libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeatureExtractor(object):
    """
    Image feature extraction with MobileNet.
    """

    def __init__(self, pooling=False, device='cpu', dtype=torch.float32):

        from torchvision import models

        self.device, self.dtype = device, dtype
        self.mobilenet = models.mobilenet_v2(pretrained=True).to(device)
        children = list(self.mobilenet.children())
        children[0][0][0] = torch.nn.Conv2d(1, 32, (3, 3), (2, 2), (1, 1), bias=False)
        self.mobilenet = nn.Sequential(*children[:-1])  # Remove the last classifier

        print(self.mobilenet)

        # average pooling
        if pooling:
            self.mobilenet.add_module('LastMaxPool', nn.MaxPool2d(4, 4))  # input: N x 1280 x 4 x 4
            self.mobilenet.add_module('LastMaxPool2', nn.MaxPool2d(2, 2))  # input: N x 1280 x 4 x 4

        self.mobilenet.eval()

    def extract_mobilenet_feature(self, img):
        num_img = img.shape[0]

        img_prepro = []
        for i in range(num_img):
            img_prepro.append(img[i])
        img_prepro = torch.stack(img_prepro).to(self.device, self.dtype)

        with torch.no_grad():
            feat = []
            process_batch = 500
            for b in range(math.ceil(num_img / process_batch)):
                feat.append(self.mobilenet(img_prepro[b * process_batch:(b + 1) * process_batch]
                                           ).squeeze(-1).squeeze(-1))  # forward and squeeze
            feat = torch.cat(feat)

            # add l2 normalization
            F.normalize(feat, p=2, dim=1)

        return feat
