import os

import numpy as np

import torch
import torch.nn as nn

import torchvision.models as models


def get_label_dict():
    # dir_path = os.path.join('../', 'data')
    # label_list = os.listdir(dir_path) # 의존성 줄이기 위해 그냥 명시적으로 적어버림
    label_list = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

    label_dict = {index: label for index, label in enumerate(label_list)}
    # print(label_dict)

    return label_dict


def get_model(device):
    device = device
    model = models.resnet50()
    model.fc = nn.Linear(in_features=2048, out_features=12)
    model.load_state_dict(torch.load('./ResNet_best.pt', map_location=device))
    model.to(device)

    return model

def preprocess_image(image, transforms):
    temp_image = np.array(image)
    
    image = transforms(image=temp_image)['image']                             
    image = image.float()
    # image = image.cuda()
    image = image.unsqueeze(0) #I don't know for sure but Resnet-50 model seems to only
                               #accpets 4-D Vector Tensor so we need to squeeze another
    
    return image