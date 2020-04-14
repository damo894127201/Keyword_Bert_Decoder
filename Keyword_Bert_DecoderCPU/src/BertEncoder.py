# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 8:42
# @Author  : Weiyang
# @File    : BertEncoder.py

# ---------------------------------------------------------------
# Encoder: Bert
# ---------------------------------------------------------------

import torch.nn as nn

class BertEncoder(nn.Module):

    def __init__(self,BertModel):
        super(BertEncoder,self).__init__()
        self.BertModel = BertModel # BertModel用于获取单词词向量和文档的句向量

    def forward(self,source):
        '''
        :param source: [batch_size,max_time_step]
        :return: 返回各个时刻 BERT模型 的输出以及 整个序列的句向量
        '''
        # 通过Bert获取词向量序列,维度为[batch_size,max_time_step,embedding_size]
        # sentenceVector: [batch_size,embedding_size]
        wordEmbeddings,sentenceVector = self.BertModel(source)
        # [batch_size,max_time_step,embedding_size]
        wordEmbeddings = wordEmbeddings.detach() # 截断梯度
        # 构造文档向量
        documentEmbedding = sentenceVector + wordEmbeddings[:,0,:] # [batch_size,embedding_size]
        documentEmbedding = documentEmbedding.unsqueeze(1) # [batch_size,1,embedding_size]
        documentEmbedding = documentEmbedding.detach() # 截断梯度

        # wordEmbeddings: [batch_size,max_time_step,embedding_size]
        # documentEmbedding: [batch_size,1,embedding_size]
        return wordEmbeddings,documentEmbedding  # 文档中各个时刻的输出，文档向量


if __name__ == '__main__':
    model = BertEncoder(None)
    print(model)