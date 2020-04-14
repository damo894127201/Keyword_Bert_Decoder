# -*- coding: utf-8 -*-
# @Time    : 2020/4/11 12:28
# @Author  : Weiyang
# @File    : Decoder.py

# -----------------------------------------------
# 综合解码器，包含三块：
# 1. TransformerDecoder生成模块
# 2. Copy Attention Distribution 模块
# 3. Keyword Attention Distribution 模块
# 4. Switching Network模块
# -----------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from TransformerDecoder import TransformerDecoder
from CopyAttentionDistribution import CopyAttentionDistribution
from KeywordAttentionDistribution import KeywordAttentionDistribution


class Decoder(nn.Module):

    def __init__(self,output_size,attention_size,embedding_size,feedforward_hidden_size,num_heads,num_blocks,keep_prob):
        super(Decoder,self).__init__()
        self.output_size = output_size # 输出词包的大小
        self.attention_size = attention_size  # Q,K,V向量的维度
        self.embedding_size = embedding_size  # TransformerDecoder输入向量的维度或Bert模型输出的词向量维度
        self.feedforward_hidden_size = feedforward_hidden_size  # 前馈神经网络的维度
        self.num_heads = num_heads  # 多头注意力的个数
        self.num_blocks = num_blocks  # 堆叠的Transformer的个数
        self.keep_prob = keep_prob  # Dropout的比例
        self.drop_out = nn.Dropout(self.keep_prob)
        self.TransformerDecoder = TransformerDecoder(attention_size=self.attention_size,embedding_size=self.embedding_size,
                                                     feedforward_hidden_size=self.feedforward_hidden_size,
                                                     num_heads=self.num_heads,num_blocks=self.num_blocks,keep_prob=self.keep_prob)
        # TransformerDecoder模块的输出在输入进softmax前经过的线性层，用来生成预测结果
        self.Linear = nn.Linear(in_features=self.embedding_size,out_features=self.output_size)
        # Copy Attention Distribution 模块
        self.CopyLayer = CopyAttentionDistribution(hidden_size=self.embedding_size,embedding_size=self.embedding_size)
        # Keyword Attention Distribution 模块
        self.KeywordLayer = KeywordAttentionDistribution(hidden_size=self.embedding_size,embedding_size=self.embedding_size)
        # 计算 Decoder 对三种生成预测结果方式的偏好
        self.SwitchingNetwork_s = nn.Linear(in_features=self.embedding_size,out_features=3)
        self.SwitchingNetwork_D = nn.Linear(in_features=self.embedding_size,out_features=3)

    def forward(self,source,target,bert_inputs,documentEmbedding,keywordEmbeddings,CopyCoverageVector,KeywordCoverageVector,flag=True):
        '''
        :param source: 输入序列[batch_size,max_time_step]
        :param target: 当前已解码序列,[batch_size,time_step,embedding_size]
        :param bert_inputs: 输入序列经过BERT处理后的各个时刻的向量表示,[batch_size,max_time_step,embedding_size]
        :param documentEmbedding: 输入序列经过BERT处理后的句向量或文档向量,[batch_size,1,embedding_size]
        :param keywordEmbeddings: 关键词序列经过BERT处理后的句向量表示,[batch_size,max_words_num,embedding_size]

        覆盖向量,用于记录Encoder各个时刻的注意力权重累计和,作用是抑制Decoder关注那些已被关注过的位置
        :param CopyCoverageVector: [batch_size,max_time_step,1],用以记录解码阶段输入序列中各个位置Copy注意力权重的累加值
        :param KeywordCoverageVector: [batch_size,max_words_num,1],用以记录解码阶段关键词序列中各个单词Copy注意力权重的累加值

        :param flag: flag=True,表示训练阶段，采用dropout;flag=False,表示预测阶段，不采用
        :return: 返回当前时刻的解码输出,以及该解码输出所属的模块类型，即是由哪一部分生成的
        '''
        # 训练阶段
        if flag == True:
            target = self.drop_out(target)
        # ----------------------------------    TransformerDecoder模块  ----------------------------------------
        # 当前TransformerDecoder的输出: [batch_size,1,embedding_size]
        current_hidden_state = self.TransformerDecoder(target)
        # TransformerDecoder模块的预测输出层: [batch_size,1,output_size]
        transformer_output = self.Linear(current_hidden_state)
        # softmax 层：[batch_size,1,output_size]
        transformer_output = F.softmax(transformer_output,dim=2)
        # [batch_size,output_size]
        transformer_output = transformer_output.squeeze(1)

        # ---------------------------------   Copy Attention Distribution 模块 ----------------------------------
        # [batch_size,output_size],[batch_size,max_time_step,1]
        Proba_copy,CopyCoverageVector,CopyCoverageLoss = self.CopyLayer(current_hidden_state,bert_inputs,source,self.output_size,CopyCoverageVector)

        # ---------------------------------   Keyword Attention Distribution 模块 --------------------------------
        # [batch_size,max_words_num],[batch_size,max_words_num,1]
        Proba_keyword,KeywordCoverageVector,KeywordCoverageLoss = self.KeywordLayer(current_hidden_state,keywordEmbeddings,KeywordCoverageVector)

        # ---------------------------------   Switching Network   ------------------------------------------------
        # [batch_size,1,3]
        switching_hidden = self.SwitchingNetwork_s(current_hidden_state)
        # [batch_size,1,3]
        switching_document = self.SwitchingNetwork_D(documentEmbedding)
        # [batch_size,1,3]
        # 分别表示选择门对：TransformerDecoder生成模块、Copy Attention Distribution 模块 和 Keyword Attention Distribution 模块的偏好
        switchingProba = F.softmax(switching_hidden+switching_document,dim=2)

        # ---------------------------------------   生成预测结果模块  --------------------------------------------
        # TransformerDecoder生成模块,各个单词最终的生成概率
        # [batch_size,output_size]
        transformer_output = transformer_output * (switchingProba[:,:,0])

        # Copy Attention Distribution 模块,各个单词最终的拷贝概率
        # [batch_size,output_size]
        Proba_copy = Proba_copy * (switchingProba[:,:,1])

        # Keyword Attention Distribution 模块，各个关键词最终的拷贝概率
        # [batch_size,max_words_num]
        Proba_keyword = Proba_keyword * (switchingProba[:,:,2])

        # 由于一个batch中，每条数据的预测结果可能由三个模块中任何一个生成，因此需针对每条数据单独处理
        # 这部分工作，我们放在另一个函数中处理，因此Decoder只返回当前时刻三个生成模块的解码结果,并不进行比较
        # [batch_size,output_size],[batch_size,output_size],[batch_size,max_words_num]
        decoderResult = (transformer_output,Proba_copy,Proba_keyword)

        # CoverageLoss
        CoverageLoss = CopyCoverageLoss + KeywordCoverageLoss

        return decoderResult,CopyCoverageVector,KeywordCoverageVector,CoverageLoss