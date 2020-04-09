import pandas as pd

data = pd.read_csv('data/label.csv', header=None)
label = data.as_matrix()
y = label[:]

print(y.shape)
print(y)