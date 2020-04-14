# -*- coding: utf-8 -*-
# @Time    : 2020/4/10 19:47
# @Author  : Weiyang
# @File    : Multihead_attention.py

# ------------------------------------------------------------
# 多头注意力机制,参考论文《Attention is all your need》
# 使用Transformer的编码器作为我们模型的解码器
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class Multihead_attention(nn.Module):

    def __init__(self,attention_size,embedding_size,num_heads):
        super(Multihead_attention,self).__init__()
        self.attention_size = attention_size # Q,K,V三个向量的维度
        self.num_heads = num_heads # 多头注意力的个数
        self.embedding_size = embedding_size # 词向量的维度,用于计算Scaled Dot-Product Attention
        # 计算多头注意力的Q,K,V三个向量,这里我们是先集中计算多个Q,K,V，之后再拆分
        self.Q_All = nn.Linear(in_features=self.embedding_size,out_features=self.attention_size*self.num_heads)
        self.K_All = nn.Linear(in_features=self.embedding_size,out_features=self.attention_size*self.num_heads)
        self.V_All = nn.Linear(in_features=self.embedding_size,out_features=self.attention_size*self.num_heads)
        # 将拼接后的多个注意力头的Z向量重新压缩回输入的尺寸大小
        self.Z_Condense = nn.Linear(in_features=self.attention_size*self.num_heads,out_features=self.embedding_size)

    def forward(self,inputs):
        '''
        :param inputs: Transformer模块的输入,[batch_size,max_time_step,embedding_size]
        :param inputs_len: Tranformer模块的输入
        :return: 返回 Z vector
        '''
        batch_size = inputs.size(0) # 当前批次大小
        # 计算多头注意力的多对Q,K,V
        # [batch_size,max_time_step,attention_size*num_heads]
        Q_Matrixs = self.Q_All(inputs)
        # [batch_size,max_time_step,attention_size*num_heads]
        K_Matrixs = self.K_All(inputs)
        # [batch_size,max_time_step,attention_size*num_heads]
        V_Matrixs = self.V_All(inputs)

        # 拆分成多个q,k,v,再在第一维上下拼接
        # Q_Matrixs:[num_heads*batch_size,max_time_step,attention_size]
        # K_Matrixs:[num_heads*batch_size,max_time_step,attention_size]
        # V_Matrixs:[num_heads*batch_size,max_time_step,attention_size]
        Q_Matrixs = torch.cat(torch.split(Q_Matrixs,self.num_heads,dim=2),dim=0)
        K_Matrixs = torch.cat(torch.split(K_Matrixs,self.num_heads,dim=2),dim=0)
        V_Matrixs = torch.cat(torch.split(V_Matrixs,self.num_heads,dim=2),dim=0)

        # 将K_Matrixs维度转换下,[num_heads*batch_size,attention_size,max_time_step]
        K_Matrixs = K_Matrixs.permute(0,2,1)

        # Dot-Product,score = q*k,[num_heads*batch_size,max_time_step,max_time_step]
        # 其中，每一行代表某个注意力头中某个时刻与同属某个注意力头的其它所有时刻的q*k点积结果
        scores = torch.bmm(Q_Matrixs,K_Matrixs)

        # scaled,[num_heads*batch_size,max_time_step,max_time_step]
        scores /= self.attention_size ** 0.5

        # softmax,[num_heads*batch_size,max_time_step,max_time_step]
        scores = F.softmax(scores,dim=2)

        # 同时计算多个注意力头的Z向量: [num_heads*batch_size,max_time_step,attention_size]
        Z = torch.bmm(scores,V_Matrixs)

        # 将多个注意力头的z向量横向拼接: [batch_size,max_time_step,attention_size*num_heads]
        Z = Z.view(batch_size,-1,self.attention_size*self.num_heads).contiguous()

        # 将多个注意力头拼接后的Z向量重新压缩回输入的大小
        Z = self.Z_Condense(Z) # [batch_size,max_time_step,embedding_size]

        return Z

if __name__ == '__main__':
    model = Multihead_attention(5,5,2)
    inputs = torch.ones((3,6,5))
    result = model(inputs)
    print(result)
    print(result.size())
