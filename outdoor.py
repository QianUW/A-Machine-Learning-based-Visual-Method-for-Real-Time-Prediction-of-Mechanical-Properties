import os
import cv2
import numpy as np
pd = np.array([5,5])
for i in os.listdir('./fig2/'):
    img = cv2.imread(i,1)
    img = cv2.copyMakeBorder(img,5,5,5,5,cv2.BORDER_CONSTANT,value=[255,255,255])
    cv2.imshow(i,img)
    cv2.waitKey(0)