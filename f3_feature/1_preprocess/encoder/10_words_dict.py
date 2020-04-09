import tensorflow as tf


"""
字符串转为ID
============
"""

string = "今天的天气不错"

set_string = set(string)

len_set = len(set_string)

word2id = dict(zip(set_string, range(len_set)))

print(word2id)

print([word2id[s] for s in string])


"""
列表转ID
=========
"""

l_class = ['财经', '时政', '教育', '游戏', '体育', '娱乐', '时尚', '家居', '科技', '房产']

d_class = {item: idx for idx, item in enumerate(l_class)}

print(d_class)
