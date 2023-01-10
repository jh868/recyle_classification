from padding import padding
import cv2

img = cv2.imread('./data/brown-glass/brown-glass (1).jpg')
print(img.shape)
img = padding(img, 256)
cv2.imwrite('03.png', img)
