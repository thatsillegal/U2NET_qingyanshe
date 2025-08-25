from __future__ import print_function, division
from skimage import io
import numpy as np
from torch.utils.data import Dataset
import cv2
import albumentations as A

# 继承 Dataset 类只有两个硬性要求：
# 必须实现 __len__() - 返回数据集大小
# 必须实现 __getitem__() - 返回单个样本
class MyDataset(Dataset):
    def __init__(self,img_name_list,lbl_name_list,transform=None):
        """
        img_name_list: 图像路径列表
        lbl_name_list: 标签路径列表，可为空
        transform: albumentations 增强管道
        """
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform

    # 这是 torch.utils.data.Dataset 必须实现的方法之一
    def __len__(self):
        return len(self.image_name_list)

    # 这是 Dataset 的核心逻辑
    def __getitem__(self,idx):
        image = cv2.imread(self.image_name_list[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if(0==len(self.label_name_list)):
            label_3 = np.zeros(image.shape)
        else:
            label_3 = io.imread(self.label_name_list[idx])

        mask = np.zeros(label_3.shape[0:2])
        # 如果是 3 通道，取第 0 个通道作为标签
        if(3==len(label_3.shape)):
            mask = label_3[:,:,0]
        # 如果是 2 通道（灰度图），直接用
        elif(2==len(label_3.shape)):
            mask = label_3

        # 保证维度数都为 3
        if(3==len(image.shape) and 2==len(mask.shape)):
            mask = mask[:,:,np.newaxis]
        elif(2==len(image.shape) and 2==len(mask.shape)):
            image = image[:,:,np.newaxis] # newaxis 是 NumPy 中用来增加数组维度的常量，它实际上就是 None 的别名
            mask = mask[:,:,np.newaxis]

        # 返回的是一个字典(dictionary)，包含三个键值对
        sample = {'image':image, 'mask':mask}

        if self.transform:
            transformed = self.transform(**sample) # albumentations 会对同一个字典的图像做同一个变换
            image = transformed['image']
            mask = transformed['mask']

        return image, mask

#==========================dataset load==========================

class NormalizeSeparate(A.DualTransform):
    def __init__(self,
                 always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        # 内置两个 Normalize
        self.normalize_rgb = A.Normalize(mean=(0.485,0.456,0.406),
                                         std=(0.229,0.224,0.225))
        self.normalize_gray = A.Normalize(mean=(0.5,), std=(0.5,))

    def apply(self, img, **params):
        img = img.astype(np.float32)  # 保持 float32
        return self.normalize_rgb(image=img)["image"]

    def apply_to_mask(self, mask, **params):
        return mask.astype(np.float32) / 255.0

    def get_transform_init_args_names(self):
        return ()