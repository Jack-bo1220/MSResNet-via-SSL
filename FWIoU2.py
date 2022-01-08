import numpy as np
from PIL import Image
import cv2 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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



path1 = ""
pres = os.listdir(path1)
path2 = ""
gts = os.listdir(path2)
test = Evaluator(2)
for index,term in enumerate(pres):
        pre_path = path1 + term
        gt_path=path2+term
        pre_img = cv2.imread(pre_path)
        gt_img = cv2.imread(gt_path)
        pre_img=np.asarray(pre_img)
        gt_img =np.asarray(gt_img)
        pre_img[pre_img==255]=1
        gt_img[gt_img > 127]=1
        if index==0:
            test.confusion_matrix=test._generate_matrix(gt_img,pre_img)
        else:
            test.add_batch(gt_img,pre_img)
        
val=test.Frequency_Weighted_Intersection_over_Union()
val2=test.Mean_Intersection_over_Union()
val3=test.Pixel_Accuracy()
print("Overall FWIoU:",val)
print("MIou: ",val2)
print("ACC: ",val3)
