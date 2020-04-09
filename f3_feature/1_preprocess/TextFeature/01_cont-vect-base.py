#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
==============================
文本向量化方法1
==============================
统计词频
"""
print(__doc__)
import sklearn.feature_extraction.text as t2v

text = ['纽约市 初步  初步 迹象 显示 初步',
        '初步 迹象 显示 这是 蓄意',
        '也 无 明确 证据 显示 迹象']

"""
文本向量化
参数：
    max_features： 字典中保留的词的个数, 对所有关键词的term frequency进行降序排序，只取前max_features个作为关键词集
    ngram_range=(1, 3)：构造新的词向量的组合，表示选取前后1到3个词做为组合方式
    max_df：忽略文档频率严格高于给定阈值（语料库特定停止词）的术语
    min_df： 忽略文档频率严格低于给定阈值的术语。
    token_pattern: 单词正则化提取方式, 默认参数为r"(?u)\b\w\w+\b"匹配长度≥2的单词, 需更改；
    encodeing：默认utf-8
    vocabulary: 指定词典
    decode_error：为ignore将会忽略解码错误
属性：
    vocabulary_：词汇表；字典型
    get_feature_names()：所有文本的词汇；列表型
    stop_words_：返回停用词表
"""
vectorizer = t2v.CountVectorizer()
"""
# fit，建立字典，所以测试集不能用fit，而是直接使用训练集的词典转换
# transform,计算词频
# 返回稀疏矩阵
"""
vectors = vectorizer.fit_transform(text)
'''todense转化为稠密矩阵'''
print("单词向量:\n", vectors.todense())
print("字典", vectorizer.vocabulary_)
