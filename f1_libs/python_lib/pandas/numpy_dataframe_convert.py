import numpy as np
import pandas as pd

arr = np.random.rand(4, 2)
np.array([[0.66867334, 0.0496808],
          [0.24225703, 0.17014163],
          [0.37133698, 0.3160525],
          [0.76333377, 0.54704594]])

# columns
columns_new = ['one', 'two']

df = pd.DataFrame(arr, columns=columns_new)
print(df)
# pandas -> numpy
df_array = df.values
print(df_array)

# Series -> DataFrame
series = pd.Series([1, 2, 3, 4])
df = series.to_frame()
# 若Series中每个元素都是数组，则需要使用to_list()方法
l = series.to_list()
df1 = pd.DataFrame(l)
nd = np.array(l)
