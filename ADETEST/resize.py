import cv2
img = cv2.imread("ADE_val_00001046.jpg",1)
img = cv2.resize(img,(256,256))
cv2.imshow('a',img)
cv2.waitKey(0)
cv2.imwrite("./or2.jpg",img)