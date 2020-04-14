# -*- coding: utf-8 -*-
# @Time    : 2020/4/11 11:19
# @Author  : Weiyang
# @File    : TransformerDecoder.py

# ---------------------------------------------------------------------------------------------------------
# TransformerDecoder: 实现了多个Transformer Encoder相互堆叠
# 这里的TransformerDecoder的子块用的是Transformer的Encoder，但解码时的方式采用的Transformer的Decoder的方式，
# 即当前时刻看不到后面时刻的内容，只能一步一步解码。
# ----------------------------------------------------------------------------------------------------------

import torch.nn as nn
from Multihead_attention import Multihead_attention
from LayerNorm import LayerNorm

class TransformerDecoder(nn.Module):

    def __init__(self,attention_size,embedding_size,feedforward_hidden_size,num_heads,num_blocks,keep_prob):
        super(TransformerDecoder,self).__init__()
        self.attention_size = attention_size # Q,K,V向量的维度
        self.embedding_size = embedding_size # TransformerDecoder输入向量的维度
        self.feedforward_hidden_size = feedforward_hidden_size # 前馈神经网络的维度
        self.num_heads = num_heads # 多头注意力的个数
        self.num_blocks = num_blocks # 堆叠的Transformer的个数
        self.keep_prob = keep_prob # Dropout的比例
        self.Multihead_attention = Multihead_attention(attention_size=self.attention_size,embedding_size=self.embedding_size,num_heads=self.num_heads)
        # 前馈神经网络模块的输入层网络
        self.FeedforwardNetworkInputLayer = nn.Linear(in_features=self.embedding_size,out_features=self.feedforward_hidden_size)
        # 前馈神经网络模块的输出层网络
        self.FeedforwardNetworkOutputLayer = nn.Linear(in_features=self.feedforward_hidden_size, out_features=self.embedding_size)
        self.drop_out = nn.Dropout(self.keep_prob)

    def forward(self,inputs,flag=True):
        '''
        :param inputs: [batch_size,max_time_step,embedding_size]
        :param flag: flag=True,表示训练阶段，采用dropout;flag=False,表示预测结果，不采用dropout
        :return: TransformerDecoder当前时刻的解码结果,[batch_size,embedding_size]
        '''
        if flag == True:
            inputs = self.drop_out(inputs)
        # Blocks
        for i in range(self.num_blocks):
            # 多头注意力机制
            # Z: [batch_size,max_time_step,embedding_size]
            Z = self.Multihead_attention(inputs)

            # 残差网络
            inputs += Z
            # LayerNorm: [batch_size,max_time_step,embedding_size]
            inputs = LayerNorm(inputs)

            # FeedForward Network
            residual = self.FeedforwardNetworkInputLayer(inputs) # [batch_size,max_time_step,feedforward_hidden_size]
            residual = self.FeedforwardNetworkOutputLayer(residual) # [batch_size,max_time_step,embedding_size]

            # 残差网络
            inputs += residual
            # LayerNorm: [batch_size,max_time_step,embedding_size]
            inputs = LayerNorm(inputs)

        # 输出当前解码时刻对应的结果，其它时刻则忽略
        current = inputs[:,-1,:]
        current = current.unsqueeze(1) # [batch_size,1,embedding_size]

        return current # [batch_size,1,embedding_size]

if __name__ == '__main__':
    import torch
    model = TransformerDecoder(5,5,6,2,3,0.5)
    inputs = torch.ones((3, 6, 5))
    result = model(inputs)
    print(result)
    print(result.size())