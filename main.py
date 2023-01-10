import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.utils.data import DataLoader
from customdata import customData
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform
train_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomFog(),
    A.RandomSnow(),
    A.RandomShadow(),
    A.RandomRotate90(),
    A.RandomBrightnessContrast(),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    ToTensorV2()
])

# dataset
train_dataset = customData('./dataset/train/', train_transform)
val_dataset = customData('./dataset/val/', val_transform)
test_dataset = customData('./dataset/test', val_transform)

# data loader
train_loader = DataLoader(train_dataset, batch_size=120, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=120, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# model call
model = models.resnet18(pretrained=True)
# print(model) (fc): Linear(in_features=512, out_features=1000, bias=True)
model.fc = nn.Linear(in_features=512, out_features=12)  # out_features 수정

model.to(device)

# train
# val
# save model
