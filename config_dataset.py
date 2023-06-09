import glob
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class CustomDataset(Dataset):
    
    def __init__(self, root, train=True, transforms=None):
        self.root = root
        self.train = train
        self.transforms = transforms
        self.imgs = sorted(glob.glob(root+'/*.png'))
        
        if train:
            self.boxes = sorted(glob.glob(root+'/*.txt'))

    def parse_boxes(self, box_path):
        with open(box_path, 'r') as file:
            lines = file.readlines()

        boxes = []
        labels = []

        for line in lines:
            values = list(map(float, line.strip().split(' ')))
            class_id = int(values[0])
            x_min, y_min = int(round(values[1])), int(round(values[2]))
            x_max, y_max = int(round(max(values[3], values[5], values[7]))), int(round(max(values[4], values[6], values[8])))

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id)

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        height, width = img.shape[0], img.shape[1]

        if self.train:
            box_path = self.boxes[idx]
            boxes, labels = self.parse_boxes(box_path)
            labels += 1 # Background = 0

            if self.transforms is not None:
                transformed = self.transforms(image=img, bboxes=boxes, labels=labels)
                img, boxes, labels = transformed["image"], transformed["bboxes"], transformed["labels"]
                
            return img, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

        else:
            if self.transforms is not None:
                transformed = self.transforms(image=img)
                img = transformed["image"]
            file_name = img_path.split('/')[-1]
            return file_name, img, width, height

    def __len__(self):
        return len(self.imgs)
    




def get_train_transforms(args):
    return A.Compose([
        A.Resize(args.image_size, args.image_size),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))



def get_test_transforms(args):
    return A.Compose([
        A.Resize(args.image_size, args.image_size),
        ToTensorV2(),
    ])
