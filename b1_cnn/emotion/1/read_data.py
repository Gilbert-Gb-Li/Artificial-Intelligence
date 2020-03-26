import keras
import numpy as np


class Data:
    def __init__(self, path):
        self.files = np.load(path)
        images = self.files['images']
        labels = self.files['labels']
        self.test_images = images[:2000]
        self.test_labels = keras.utils.to_categorical(labels[:2000])
        self.train_images = images[2001:]
        self.train_labels = keras.utils.to_categorical(labels[2001:])

    def get_random_train_batch(self, batch_size):
        random_index = np.random.randint(0, len(self.train_labels), [batch_size])
        return self.train_images[random_index], self.train_labels[random_index]

    def test_data_iterator(self, batch_size):
        for i in range(1, len(self.test_labels) // batch_size + 1):
            yield self.test_images[(i - 1) * batch_size:i * batch_size], self.test_labels[
                                                                         (i - 1) * batch_size:i * batch_size]

    def get_batch_test_data(self, batch_size):
        iterator = self.test_data_iterator(batch_size)
        return next(iterator)


if __name__ == '__main__':
    data = Data('data.npz')
    for i in range(42):
        print(data.get_batch_test_data(50))
