# -*- coding: utf-8 -*-
# @Time    : 2020/4/11 10:40
# @Author  : Weiyang
# @File    : LayerNorm.py

# -----------------------------------------------------------------
# LayerNorm层,参考Jimmy Lei Ba等人的论文《Layer Normalization》而实现
# LayerNorm与BatchNormalization的区别：
# under layer normalization, all the hidden units in a layer share the same normalization terms
# µ and σ, but different training cases have different normalization terms. Unlike batch normalization,
# layer normaliztion does not impose any constraint on the size of a mini-batch and it can be used in
# the pure online regime with batch size 1
# ----------------------------------------------------------------

import torch
from torch.autograd import Variable

def LayerNorm(inputs,epsilon = 1e-8):
    '''
    :param inputs: [batch_size,max_time_step,embedding_size]
    :param epsilon: 一个小量,防止方差为0
    :return: [batch_size,max_time_step,embedding_size]
    '''
    embedding_size = inputs.size(2)
    # 每条数据在当前层的各units单元的方差和均值
    # unit_var:[batch_size,max_time_step,1]
    # unit_mean:[batch_size,max_time_step,1]
    unit_var,unit_mean = torch.var_mean(inputs,dim=2,keepdim=True)

    # beta,gama用于平滑处理layerNorm后的数据: [embedding_size]
    beta = Variable(torch.zeros(embedding_size),requires_grad=True).cuda()
    gama = Variable(torch.ones(embedding_size),requires_grad=True).cuda()

    # 标准化: (x-mean)/sigma
    inputs = (inputs-unit_mean) / ((unit_var+epsilon) ** 0.5)
    # 平滑处理
    inputs = gama * inputs + beta

    return inputs

if __name__ == '__main__':
    inputs = torch.ones((5,3,6))*3
    result = LayerNorm(inputs)
    print(inputs)
    print(result)
    print(inputs.size())
    print(result.size())
