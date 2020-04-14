# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 16:40
# @Author  : Weiyang
# @File    : pad.py

# ----------------------------------------------------------------------
# torch.utils.data.DataLoader的collate_fn回调函数
# 用于对一个batch进行处理：
# 1. 填充到等长
# 2. 返回batch各条数据实际长度，并逆序(target随source排序)
# 3. 对应的是MyDataSet数据集
# 4. 将每个batch中每篇文章的关键词的个数以及关键词的长度处理为等长
# ----------------------------------------------------------------------

from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np

def pad(batch):
    padding_value = 0 # 填充值索引
    source,target,keyword = [],[],[] # 注意，keyword是一个三维列表，[[[id,id,..],[id,id,...],...],...]
                                     # 第一维是当前batch中文章的个数，第二维是文章的关键词个数，第三维是每个关键词的字符数
    source_lens = []
    target_lens = []
    keyword_lens = [] # 存储每篇文章的关键词个数
    every_keyword_lens = [] # 存储每篇文章关键词的最大长度，用于将关键词填充到等长
    for x,y,z in batch:
        source.append(x)
        target.append(y)
        keyword.append(z)
        source_lens.append(len(x))
        target_lens.append(len(y))
        keyword_lens.append(len(z))
        every_keyword_lens.append(max([len(i) for i in z]))

    # 当前batch中每篇文章最大的关键词个数
    max_words_num = max(keyword_lens)
    # 当前batch中所有关键词的最大长度
    max_words_len = max(every_keyword_lens)

    pad_id = 0 # 用于填充每个关键词的空白字符
    PAD_ID = [0]*max_words_len # 用于填充的空白关键词,长度为max_words_len，即[PAD]符号

    # 将每篇文章的关键词个数填充到相同的个数
    temp = []
    for word in keyword:
        new_word = []
        # 如果当前关键词的长度小于max_words_len，则填充
        for chrs in word:
            if len(chrs) < max_words_len:
                chrs.extend([pad_id]*(max_words_len-len(chrs)))
            new_word.append(chrs)
        # 将每篇文章的关键词个数扩充到等长
        if len(new_word) < max_words_num:
            new_word.extend([PAD_ID]*(max_words_num-len(word)))
        temp.append(new_word)
    keyword = temp
    # 对source长度倒排，target和keyword随其排序
    source_lens = np.array(source_lens)
    descendingOrders = np.argsort(-source_lens)
    source = np.array(source)[descendingOrders].tolist()
    target = np.array(target)[descendingOrders].tolist()
    keyword = np.array(keyword)[descendingOrders].tolist()
    source_lens = torch.tensor(np.sort(source_lens)[::-1].tolist()) # 相应的长度从大到小排序
    target_lens = np.array(target_lens)[descendingOrders].tolist() # target端的长度随source而排序
    keyword_lens = np.array(keyword_lens)[descendingOrders].tolist() # keyword端的长度随source而排序
    # 转为torch.tensor
    source = [torch.tensor(seq) for seq in source]
    target = [torch.tensor(seq) for seq in target]
    keyword = torch.tensor(keyword)
    # 填充,batch*max_time_step
    source = pad_sequence(source,batch_first=True,padding_value=padding_value)
    target = pad_sequence(target,batch_first=True,padding_value=padding_value)
    return source,target,source_lens,target_lens,keyword,keyword_lens