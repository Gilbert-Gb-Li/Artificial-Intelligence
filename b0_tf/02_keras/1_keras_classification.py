import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import pandas as pd
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

# Loading the dataset, dataset is dataframe
dataset = pd.read_csv('../../data/iris.data.csv')
dataset = pd.get_dummies(dataset, columns=['Species'])
values = list(dataset.columns.values)
# print(dataset.info())
# print(values[0:5])
y = dataset[values[-3:]]
# print(y)
y = np.array(y, dtype='float32')
X = dataset[values[1:-3]]
X = np.array(X, dtype='float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)

# print("shape")
# print(X_train.shape)
# print(X_train[0:1])

model = Sequential()
model.add(Dense(16, input_shape=(3,)))
model.add(Activation("sigmoid"))
# model.add(Dense(50))
# model.add(Activation("sigmoid"))
# model.add(Dense(5))
# model.add(Activation("sigmoid"))
model.add(Dense(3))
model.add(Activation("softmax")) 
model.summary()
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# loss = model.evaluate(X_test, y_test, verbose=1)
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy = {:.2f}".format(accuracy))