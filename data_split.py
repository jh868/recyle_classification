import os
import glob
from sklearn.model_selection import train_test_split
import shutil

path = './data/'
dataset_path = './dataset/'
image_folder_list = os.listdir(path)
os.makedirs(dataset_path, exist_ok=True)

# 각 이미지폴더 마다 이미지
for image_folder in image_folder_list:
    image_path = glob.glob(os.path.join(path, image_folder, '*.jpg'))
    print(image_path)
    train, temp = train_test_split(image_path, test_size=0.2)
    val, test = train_test_split(temp, test_size=0.5)

    os.makedirs(dataset_path + 'train/' + image_folder, exist_ok=True)
    os.makedirs(dataset_path + 'val/' + image_folder, exist_ok=True)
    os.makedirs(dataset_path + 'test/' + image_folder, exist_ok=True)

    for idx, image in enumerate(train):
        shutil.move(image, dataset_path + 'train/' + image_folder + '/' + f'{image_folder + str(idx)}.png')
    for idx, image in enumerate(val):
        shutil.move(image, dataset_path + 'val/' + image_folder + '/' + f'{image_folder + str(idx)}.png')
    for idx, image in enumerate(test):
        shutil.move(image, dataset_path + 'test/' + image_folder + '/' + f'{image_folder + str(idx)}.png')
