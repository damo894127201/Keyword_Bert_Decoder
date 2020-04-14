# -*- coding: utf-8 -*-
# @Time    : 2020/4/11 15:13
# @Author  : Weiyang
# @File    : KeywordEncoder.py

# ---------------------------------------------------------------
# KeywordEncoder: 用Bert编码每条数据的关键词序列
# 由于bert是基于字符的，因此对于关键词，我们以bert输出的句向量表示
# ---------------------------------------------------------------

import torch.nn as nn
import torch

class KeywordEncoder(nn.Module):

    def __init__(self,BertModel,embedding_size):
        super(KeywordEncoder,self).__init__()
        self.BertModel = BertModel # BertModel用于获取单词词向量和文档的句向量
        self.embedding_size = embedding_size # Bert模型的词向量维度

    def forward(self,keywords):
        '''
        :param keywords: [batch_size,max_words_num,max_word_len]
        :return: 返回各个关键词的向量表示:[batch_size,max_words_num,embedding_size]
        '''
        batch_size,max_words_num = keywords.size(0),keywords.size(1)
        # 创建tensor用于存储输出结果: [batch_size,max_words_num,embedding_size]
        sentenceVectors = torch.zeros((batch_size,max_words_num,self.embedding_size))

        # 遍历每条数据
        for i in range(batch_size):
            # 通过Bert获取关键词的向量序列,维度为[max_words_num,embedding_size]
            _,sentenceVector = self.BertModel(keywords[i])
            # [max_words_num,embedding_size]
            sentenceVectors[i] = sentenceVector
        # 截断梯度
        sentenceVectors = sentenceVectors.detach()

        return sentenceVectors  # 关键词的向量表示:[batch_size,max_words_num,embedding_size]


if __name__ == '__main__':
    model = KeywordEncoder(None)
    print(model)