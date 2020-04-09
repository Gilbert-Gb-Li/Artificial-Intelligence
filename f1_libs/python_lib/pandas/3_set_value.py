#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import pandas as pd
import numpy as np

dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=['A', 'B', 'C', 'D'])
print(df)
df.iloc[2,2] = 1111
df.loc['2013-01-03', 'D'] = 2222

df.A[df.A>70] = 0
df.A[df.A<=70] = 1

df['F'] = np.nan
df['G']  = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130101', periods=6))
print(df)

#计算各列数据总和并作为新列添加到末尾 x[0]
df['Col_sum'] = df.apply(lambda x: x[0] + x[1], axis=1)
print(df)
#计算各行数据总和并作为新行添加到末尾
df.loc['Row_sum'] = df.apply(lambda x: x.sum())
print(df)