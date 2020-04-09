import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

method = KMeans(n_clusters=8)
image = cv2.imread("paris.jpg")
# [高，宽，通道]
h, w, c = np.shape(image)
image_2d = np.reshape(image, [-1, c])
method.fit(image_2d)
y = method.predict(image_2d)
y_img = np.reshape(y, [h, w])
plt.matshow(y_img)
plt.show()
