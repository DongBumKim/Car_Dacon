#!/usr/bin/env python
# coding: utf-8

# ## Import
import warnings
warnings.filterwarnings(action='ignore')
import torch
from torch.utils.data import  DataLoader
from sklearn.model_selection import train_test_split
import random

from batch_engine import train, inference
from model import build_model
from util import determine_all_seed, draw_boxes_on_image, collate_fn
from config import argument_parser
from config_dataset import CustomDataset, get_train_transforms, get_test_transforms
from loss import FasterRCNNLoss
import torchvision.models.detection as models

# ## Hyperparameter Setting
parser = argument_parser()
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(device)

# ## Fixed Random-Seed
determine_all_seed(args.random_seed)
    
# 파일 경로 설정
#image_file = '/home/briankim/Development/Dataset/car_classification/train/syn_00000.png'
#annotation_file = '/home/briankim/Development/Dataset/car_classification/train/syn_00000.txt'

# 함수 실행
#draw_boxes_on_image(image_file, annotation_file)


# ## Custom Dataset



train_dataset = CustomDataset('/home/briankim/Development/Dataset/car_classification/debug_train', train=True, transforms=get_train_transforms(args))
test_dataset = CustomDataset('/home/briankim/Development/Dataset/car_classification/test', train=False, transforms=get_test_transforms(args))

print(len(train_dataset))
train_ratio = 0.8  # Ratio of data to use for training (80%)
train_size = int(train_ratio * len(train_dataset))
indices = list(range(len(train_dataset)))
random.shuffle(indices)

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_subset = torch.utils.data.Subset(train_dataset, train_indices)
val_subset = torch.utils.data.Subset(train_dataset, val_indices)
print("Dataset")

# DataLoader 생성
train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)




# ## Train & Validation
print("Build Model")

model = build_model(args)




optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

criterion = FasterRCNNLoss()

infer_model = train(args, model, train_loader, val_loader, optimizer, scheduler, device, criterion)


# ## Inference & Submission


inference(args, infer_model, test_loader, device)
