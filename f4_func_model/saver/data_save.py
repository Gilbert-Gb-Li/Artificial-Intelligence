import numpy as np

"""-------------------
# 1.
# 使用numpy 保存数据文件
----------------------"""
a = np.array([1, 2, 3, 4, 5])

# 保存到 outfile.npy 文件上
np.save('outfile.npy', a)

# 保存到 outfile2.npy 文件上，如果文件路径末尾没有扩展名 .npy，该扩展名会被自动加上
np.save('outfile2', a)

b = np.load('outfile.npy')
print(b)

import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.arange(0, 1.0, 0.1)
c = np.sin(b)

'''
# c 使用了关键字参数 sin_array
# 保存为字典
'''
np.savez("runoob.npz", a, b, sin_array=c)

'''# 字典，无法使用元组'''
r = np.load("runoob.npz")
print(r.files)  # 查看各个数组名称
print(r["arr_0"])  # 数组 a
print(r["arr_1"])  # 数组 b
'''# 数组 c'''
print(r["sin_array"])

"""----------------
# 2.
# 使用pickle保存数据
# 另外：
    dumps和loads可以将数据保存成字符串
-------------------"""

import pickle

# 序列化到文件
obj = 123, "abcdedf", ["ac", 123], {"key": "value", "key1": "value1"}
print(obj)
# wb 读写到二进制文件
f = open("./a", 'wb')
pickle.dump(obj, f)
f.close()
f = open("./a", 'rb')
print(pickle.load(f))
f.close()
