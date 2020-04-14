# -*- coding: utf-8 -*-
# @Time    : 2020/4/11 16:23
# @Author  : Weiyang
# @File    : KeywordAttentionDistribution.py

# -------------------------------------------------
# Copy Attention Distribution 模块
# --------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class KeywordAttentionDistribution(nn.Module):

    def __init__(self,hidden_size,embedding_size):
        super(KeywordAttentionDistribution,self).__init__()
        self.hidden_size = hidden_size # 当前隐状态的维度
        self.embedding_size = embedding_size # bert输出的词向量的维度
        # current_hidden_state
        self.Linear_s = nn.Linear(in_features=self.hidden_size,out_features=1)
        # bert_inputs
        self.Linear_b = nn.Linear(in_features=self.embedding_size,out_features=1)


    def forward(self,current_hidden_state,bert_inputs,KeywordCoverageVector):
        '''
        :param current_hidden_state: 当前TransformerDecoder输出的隐状态，[batch_size,1,hidden_size]
        :param bert_inputs: 输入的关键词序列经过BERT处理后的关键词向量表示,[batch_size,max_words_num,embedding_size]
        :param KeywordCoverageVector: [batch_size,max_words_num,1],用以记录解码阶段关键词序列中各个单词Copy注意力权重的累加值
        :return:
        '''
        assert self.hidden_size == self.embedding_size,print('TransformerDecoder的输出维度与Bert模型的词向量维度不一致!')
        max_time_step = bert_inputs.size(1) # 输入关键词的最大个数
        # 将current_hidden_state沿着第二维度复制max_words_num份，方便计算注意力权重
        # multi_current_hidden_state: [batch_size,max_words_num,hidden_size]
        multi_current_hidden_state = current_hidden_state.repeat(1,max_time_step,1)

        # [batch_size,max_words_num,1]
        multi_current_hidden_state = self.Linear_s(multi_current_hidden_state)
        # [batch_size,max_words_num,1]
        bert_inputs = self.Linear_b(bert_inputs)

        # --------------------------------------------- Coverage mechanism -----------------------------------------

        # 计算Coverage损失,用于惩罚那些注意力权重经常比较大的位置
        # coverage mechanism
        # needs an extra loss function to penalize repeatedly attending
        # to the same locations, otherwise it would be ineffective with
        # no discernible reduction in repetition

        # attention_weights: [batch_size,max_words_num,1]
        # CoverageVector: [batch_size,max_words_num,1]

        # [batch_size,max_words_num,1]
        attention_weights = torch.tanh(multi_current_hidden_state + bert_inputs + KeywordCoverageVector)
        # softmax: [batch_size,max_words_num,1]
        attention_weights = F.softmax(attention_weights,dim=1)

        # 累加注意力权重到覆盖向量中
        KeywordCoverageVector += attention_weights

        ac = torch.cat((attention_weights, KeywordCoverageVector), 2)  # [batch_size,max_words_num,2]
        CoverageLoss = torch.min(ac, dim=2)  # (min,min_index)
        CoverageLoss = torch.sum(CoverageLoss[0].data, dtype=torch.float32)

        # [batch_size,max_words_num]
        attention_weights = attention_weights.squeeze(2)

        # [batch_size,max_words_num],[batch_size,max_words_num,1],tensor标量
        return attention_weights,KeywordCoverageVector,CoverageLoss

if __name__ == '__main__':
    model = KeywordAttentionDistribution(2,2)
    current_hidden_state = torch.ones((3,1,2)) # [3,1,2]
    bert_inputs = torch.ones((3,6,2)) # [3,6,2]
    KeywordCoverageVector = torch.ones((3,6,1))
    result,CoverageVector,loss = model(current_hidden_state,bert_inputs,KeywordCoverageVector)
    print(current_hidden_state)
    print(bert_inputs)
    print(result)
    print(result.size())
    print(CoverageVector)
    print(CoverageVector.size())
    print(loss)