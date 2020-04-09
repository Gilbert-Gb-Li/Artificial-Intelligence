def process_data():
    poetry_file = '../../data/poetry.txt'
    word_dict = '../data/poetry_map.json'

    with open(word_dict, "r", encoding="utf-8") as word_dict_file:
        word2id = eval(word_dict_file.read())
    with open(poetry_file, "r", encoding="utf-8") as f:
        poetrys = [it for it in f.readlines() if len(it.strip()) > 1]
    '''-- 如果该词不在字典中，则返回字典长度为值 --'''
    get_id = lambda word: word2id.get(word, len(word2id))
    '''-------------------------
    # map函数的使用
        - 依次迭代迭代器中的元素
        - map(function, iterator)
        - 返回迭代器，需要其他方法取出
    -----------------------------'''
    poetry_vector = [list(map(get_id, poetry)) for poetry in poetrys]
    return poetry_vector, word2id


import os
def get_file_list(dirs):
    """
    dirs: 输入文件夹
    output： 获取文件夹中所有文件和文件名路径
    # root 所指的是当前正在遍历的这个文件夹的本身的地址
    # dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
    # files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
    """
    paths = []
    for root, dirs, files in os.walk(dirs):
        # print(root, dirs, files)
        for file in files:
            paths.append(root + "/" + file)
    return paths
