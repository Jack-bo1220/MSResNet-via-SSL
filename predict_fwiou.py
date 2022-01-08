import argparse
import logging
import os
import time
from xml.dom.minidom import Document
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from network import MSResNet

from utils.dataset import BasicDataset
import cv2


import numpy as np

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        #print("aasss",np.sum(self.confusion_matrix))
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)

        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def predict_img(net,
                full_img,
                device,
                scale_factor=1):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
    #
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        #output_aux,output=net(img)
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask

def get_output_filenames(input):
    in_files = input
    out_files = []
    for f in in_files:
        pathsplit = os.path.splitext(f)
        out_files.append(pathsplit[0] + '_gt.png')
    return out_files

def predict_fwiou(epoch):
    in_filePath = ''

    input = os.listdir(in_filePath)
    input.sort()
    input.sort(key=lambda x: len(x))


    in_files = []
    for i in input:
        in_files.append(in_filePath + '/' + i)
    out_filePath = ''

    if not os.path.exists(out_filePath):
        os.makedirs(out_filePath)
    out_files = get_output_filenames(input)


    net = MSResNet()

    net.n_classes = 1
    net.n_channels = 3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device=torch.device('cpu')
    model = f'./checkpoints/MSResNet_via_SSL-epoch{epoch + 1}.pth'
    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))

    for ii, fn in enumerate(in_files):
        img = Image.open(fn)
        #img0 = Image.open(fn)
        img_rows, img_cols = img.size
        img = img.resize((448, 448), Image.ANTIALIAS)
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=1,
                           device=device)

        mask = mask > 0.5

        out_fn = out_files[ii]
        out_fn = out_filePath + '/' + out_fn
        result = Image.fromarray((mask * 255).astype(np.uint8))
        result = result.resize((img_rows, img_cols), Image.ANTIALIAS)

        rows, cols = result.size
        pixel = result.load()
        # print(pixel)
        for i in range(rows):
            for j in range(cols):
                if pixel[i, j] <= 127:
                    pixel[i, j] = 0
                else:
                    pixel[i, j] = 255
        result.save(out_fn)

    path1 = ""
    pres = os.listdir(path1)
    #print(pres)
    path2 = ""
    gts = os.listdir(path2)
    test = Evaluator(2)
    # print(gts)
    for index, term in enumerate(pres):
        # print(index)
        pre_path = path1 + term
        gt_path = path2 + term
        pre_img = cv2.imread(pre_path)
        gt_img = cv2.imread(gt_path)
        pre_img = np.asarray(pre_img)
        gt_img = np.asarray(gt_img)
        pre_img[pre_img == 255] = 1
        gt_img[gt_img > 127] = 1
        if index == 0:
            test.confusion_matrix = test._generate_matrix(gt_img, pre_img)
        else:
            test.add_batch(gt_img, pre_img)


    miou =  test.Mean_Intersection_over_Union()
    del net
    return miou

if __name__ == '__main__':
    epoch=2
    fwiou=predict_fwiou(epoch)
    print(fwiou)
