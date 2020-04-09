from keras.preprocessing import image
from keras.models import Model
import numpy as np

train = []

img_path = './data/car.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
train.append(x)

img_path = './data/bird.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
train.append(x)

train = np.asarray(train, dtype=np.float32)

print(train.shape)  # (2, 224, 224, 3)
print(train[0])
