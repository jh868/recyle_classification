from padding import padding
import cv2
import glob
import os

path = ''
paper = './paper1/'
os.makedirs(paper, exist_ok=True)
image_path = glob.glob(os.path.join(path, '*.jpg'))
for idx, i in enumerate(image_path):
    img = cv2.imread(i)
    img = padding(img, 224)
    cv2.imwrite(paper + f'{idx}.jpg', img)
# img = cv2.imread()
# print(img.shape)
# img = padding(img, 256)
# cv2.imwrite('03.png', img)
