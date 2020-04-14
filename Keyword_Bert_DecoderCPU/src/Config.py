# -*- coding: utf-8 -*-
# @Time    : 2020/3/11 10:00
# @Author  : Weiyang
# @File    : Config.py

#-------------------------------------------------
# 超参
#-------------------------------------------------

class Config(object):
    vocab_size = 21128  # 输入空间：词包大小还得加上这些特殊符号："_PAD": 0, "_GO": 1, "_EOS": 2, "_UNK": 100
                        # 在BERT中,[PAD]:0,[UNK]:100,
                        # 我们用[unused1]表示GO符号,[unused1]:1
                        # 用[unused2]表示EOS符号，[unused2]:2
    embedding_size = 768 # bert词向量的维度
    attention_size = 64 # 多头自注意力中,Q,K,V三个向量的维度
    feedforward_hidden_size = 1024 # Transformer全连接层隐藏层维度
    num_heads = 8 # 注意力头的个数
    num_blocks = 6 # 堆叠的Transformer的个数
    output_size = 21128 # 输出空间,bert词包大小
    max_decoding_steps = 30
    batch_size = 8
    learning_rate = 0.0001
    num_Epochs = 20
    TeacherForcingRate = 0.8
    max_norm = 5 # 相应范数的最大值，梯度裁剪
    norm_type = 2 # 范数的类型，1表示范数L1，2表示范数L2
    use_drop_out = True # 是否使用Dropout层，此处该层设置在Decoder模块的Embedding层
                        # Dropout可放置在：可见层(输入层)、相邻隐藏层之间、隐藏层和输出层之间
    keep_prob = 0.5 # drop_out的比例通常在0.2-0.5之间，具体需要尝试
    ratio = 0.3 # GlobalCoverageLoss的比重