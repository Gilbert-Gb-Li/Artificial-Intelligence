import cv2
import numpy as np
method = cv2.ximgproc.segmentation.createGraphSegmentation()
cap = cv2.VideoCapture(1)
while True:
    ret, img = cap.read()
    seg_img = method.processImage(img)
    out_img = np.zeros_like(img, dtype=np.uint8)
    out_img[:, :, 0] = seg_img % 64 * 4
    out_img[:, :, 1] = 255 - seg_img
    out_img[:, :, 2] = seg_img % 32
    out = np.concatenate([img, out_img], axis=1)
    cv2.imshow("a", out)
    cv2.waitKey(100)
