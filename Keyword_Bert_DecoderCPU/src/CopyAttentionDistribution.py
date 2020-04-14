# -*- coding: utf-8 -*-
# @Time    : 2020/4/11 15:53
# @Author  : Weiyang
# @File    : CopyAttentionDistribution.py

# -------------------------------------------------
# Copy Attention Distribution 模块
# --------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class CopyAttentionDistribution(nn.Module):

    def __init__(self,hidden_size,embedding_size):
        super(CopyAttentionDistribution,self).__init__()
        self.hidden_size = hidden_size # 当前隐状态的维度
        self.embedding_size = embedding_size # bert输出的词向量的维度
        # current_hidden_state
        self.Linear_s = nn.Linear(in_features=self.hidden_size,out_features=1)
        # bert_inputs
        self.Linear_b = nn.Linear(in_features=self.embedding_size,out_features=1)


    def forward(self,current_hidden_state,bert_inputs,source,output_size,CopyCoverageVector):
        '''
        :param current_hidden_state: 当前TransformerDecoder输出的隐状态，[batch_size,1,hidden_size]
        :param bert_inputs: 输入序列经过BERT处理后的各个时刻的向量表示,[batch_size,max_time_step,embedding_size]
        :param source: 输入序列,[batch_size,max_time_step]
        :param output_size: 输出词包大小
        :param CopyCoverageVector: [batch_size,max_time_step,1],用以记录解码阶段输入序列中各个位置Copy注意力权重的累加值
        :return:词包中各个单词的拷贝概率，那些不在输入序列中的词的拷贝概率为0
        '''
        assert self.hidden_size == self.embedding_size,print('TransformerDecoder的输出维度与Bert模型的词向量维度不一致!')
        max_time_step = bert_inputs.size(1) # 输入文本的最大长度
        # 将current_hidden_state沿着第二维度复制max_time_step份，方便计算注意力权重
        # multi_current_hidden_state: [batch_size,max_time_step,hidden_size]
        multi_current_hidden_state = current_hidden_state.repeat(1,max_time_step,1)

        # [batch_size,max_time_step,1]
        multi_current_hidden_state = self.Linear_s(multi_current_hidden_state)
        # [batch_size,max_time_step,1]
        bert_inputs = self.Linear_b(bert_inputs)

        # --------------------------------------------- Coverage mechanism -----------------------------------------

        # 计算Coverage损失,用于惩罚那些注意力权重经常比较大的位置
        # coverage mechanism
        # needs an extra loss function to penalize repeatedly attending
        # to the same locations, otherwise it would be ineffective with
        # no discernible reduction in repetition

        # attention_weights: [batch_size,max_time_step,1]
        # CoverageVector: [batch_size,max_time_step,1]
        # 比较上述两个tensor的最后一维度的相应值，取较小值作为当前序列中当前时刻的损失
        # [batch_size,max_time_step,1]
        attention_weights = torch.tanh(multi_current_hidden_state + bert_inputs + CopyCoverageVector)
        # softmax: [batch_size,max_time_step,1]
        attention_weights = F.softmax(attention_weights,dim=1)

        # 累加注意力权重到覆盖向量中
        CopyCoverageVector += attention_weights

        ac = torch.cat((attention_weights, CopyCoverageVector), 2)  # [batch_size,max_time_step,2]
        CoverageLoss = torch.min(ac, dim=2)  # (min,min_index)
        CoverageLoss = torch.sum(CoverageLoss[0].data, dtype=torch.float32)

        # [batch_size,max_time_step]
        attention_weights = attention_weights.squeeze(2)

        # 我们需要将不同位置相同的字符的copy概率累加到一起，最后输出一个概率分布，大小为词包大小
        # 因此，那些不在输入序列中，但在词包中的字符的概率为0
        # copying a word directly from the corresponding source text based on attention distribution
        # 在解码时，我们将输入序列中各个时刻的注意力权重，作为相应时刻对应单词的copy概率
        batch_size = current_hidden_state.size(0)  # 当前batch的大小
        # 存储当前解码时刻，source端各个位置的累积的copy概率(注意力权重),其形状要与词包大小一致，使得可以输出预测结果的概率分布，便于计算交叉熵损失
        # [batch_size,output_size]
        source_position_copy_prob = torch.zeros((batch_size,output_size))
        max_time_step = source.size(1)
        # 遍历每条数据
        for i in range(batch_size):
            # 当前输入序列的ID序列
            A = source[i] # [max_time_step,]
            # 当前输入序列的copy概率(注意力权重)序列
            B = attention_weights[i]
            # 构建一个[max_time_step,output_size]的零tensor,以存储每个时刻output_size维度是否出现有A中的ID
            mask = torch.zeros((max_time_step,output_size))
            index = torch.arange(max_time_step)
            mask[index,A] = 1
            # 将A中ID在B中的概率累加起来，并存储到维度为output_size的tensor中
            C = torch.matmul(B,mask) # [output_size,]
            source_position_copy_prob[i] = C
        # [batch_size,output_size],[batch_size,max_time_step,1]
        return source_position_copy_prob,CopyCoverageVector,CoverageLoss

if __name__ == '__main__':
    model = CopyAttentionDistribution(2,2)
    source = torch.ones((3,6),dtype=torch.int32)
    output_size = 7
    current_hidden_state = torch.ones((3,1,2)) # [3,1,2]
    bert_inputs = torch.ones((3,6,2)) # [3,6,2]
    CopyCoverageVector = torch.ones((3,6,1))
    result,CoverageVector,loss = model(current_hidden_state,bert_inputs,source,output_size,CopyCoverageVector)
    print(current_hidden_state)
    print(bert_inputs)
    print(result)
    print(result.size())
    print(CoverageVector)
    print(CoverageVector.size())
    print(loss)