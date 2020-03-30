#  coding : utf-8
"""
集束搜索 beam search
--------
可以认为是维特比算法的贪心形式，在维特比所有中由于利用动态规划导致当字典较大时效率低，
而集束搜索使用beam size参数来限制在每一步保留下来的可能性词的数量。
集束搜索是在测试阶段为了获得更好准确性而采取的一种策略，在训练阶段无需使用。

假设字典为[a,b,c]，beam size选择2，则如下图有：
1：在生成第1个词的时候，选择概率最大的2个词，那么当前序列就是a或b
2：生成第2个词的时候，我们将当前序列a或b，分别与字典中的所有词进行组合，
    得到新的6个序列aa ab ac ba bb bc,然后从其中选择2个概率最高的，作为当前序列，即ab或bb
3：不断重复这个过程，直到遇到结束符为止。最终输出2个概率最高的序列。
"""

from math import log
from numpy import array
# beam search
def beam_search_decoder(data, k):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences


# define a sequence of 10 words over a vocab of 5 words
data = [[0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1]]
data = array(data)
# decode sequence
result = beam_search_decoder(data, 3)
# print result
for seq in result:
    print(seq)
