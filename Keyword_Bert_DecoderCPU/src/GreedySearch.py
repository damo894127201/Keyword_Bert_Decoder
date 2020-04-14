# -*- coding: utf-8 -*-
# @Time    : 2020/4/11 21:18
# @Author  : Weiyang
# @File    : GreedySearch.py

# ---------------------------------------------------------
# 获取当前解码时刻的预测结果，及其Embedding表示
# ---------------------------------------------------------

import torch

def GreedySearch(decoderResult,output_size,bertModel,keywords,keywordsEmbedding,flag=True):
    '''
    :param decoderResult: Decoder的输出,(transformer_output, Proba_copy, Proba_keyword)
    :param output_size: 词包大小
    :param bertModel: bert模型
    :param keywords: 关键词序列，[batch_size,max_words_num,max_word_len],关键词序列
    :param keywordsEmbedding: [batch_size,max_words_num,embedding_size]
    :param flag: flag=True,表示训练阶段;flag=False,表示预测阶段
    :return: flag=True时,返回当前解码结果的概率分布和单词Embedding, 注意,如果解码结果是由Keyword Attention Distribution 模块
             生成，则解码结果是多个char组成的单词,而我们的模型是基于char的,这会导致在计算交叉熵损失时的不匹配，因此对于生成
             的关键词的概率分布，我们将关键词中多个字符的概率设为相等，其它词包中的字符概率设为0；
             flag=False时,返回当前解码结果的单词ID序列和单词Embedding
    '''
    batch_size,embedding_size = keywordsEmbedding.size(0),keywordsEmbedding.size(2)
    # 获取三个模块生成的char概率分布
    # [batch_size,output_size],[batch_size,output_size],[batch_size,max_word_num]
    transformer_output, Proba_copy, Proba_keyword = decoderResult
    # 分别选出三个模块中概率最大的单词，然后取这三个模块中概率最大单词作为最终的预测结果
    # value: [batch_size,1] , index: [batch_size,1]
    transformer_value, transformer_index = transformer_output.topk(1)
    Proba_copy_value, Proba_copy_index = Proba_copy.topk(1)
    Proba_keyword_value, Proba_keyword_index = Proba_keyword.topk(1)

    # 定义三个tensor，用以指示不同模块
    type0 = torch.tensor(0)  # TransformerDecoder生成模块
    type1 = torch.tensor(1)  # Copy Attention Distribution 模块
    type2 = torch.tensor(2)  # Keyword Attention Distribution 模块
    PAD_ID = torch.tensor(0) # [PAD]的ID

    if flag == True:
        # 训练阶段
        batch_word_prob_Distribution = torch.zeros((batch_size,output_size)) # 存储当前预测结果的概率分布
        # 当前解码结果对应的Embedding
        batch_word_Embedding = torch.zeros((batch_size, 1, embedding_size))
        # 遍历每一条数据，获取其解码结果
        for i in range(batch_size):
            prob_list = torch.tensor([transformer_value[i][0], Proba_copy_value[i][0], Proba_keyword_value[i][0]])
            # 获取概率最大的模块,typeIndex: tensor([value])
            _, typeIndex = prob_list.topk(1)
            if typeIndex[0] == type0:
                # 表示TransformerDecoder生成模块生成的单词概率最大
                batch_word_prob_Distribution[i] = transformer_output[i]
                wordEmbedding, _ = bertModel(transformer_index[i].unsqueeze(0)) # [1,1,embedding_size]
                batch_word_Embedding[i] = wordEmbedding[0]  # [1,embedding_size]
            elif typeIndex[0] == type1:
                # 表示Copy Attention Distribution 模块 生成的单词概率最大
                batch_word_prob_Distribution[i] = Proba_copy[i]
                wordEmbedding, _ = bertModel(Proba_copy_index[i].unsqueeze(0))  # [1,1,embedding_size]
                batch_word_Embedding[i] = wordEmbedding[0]  # [1,embedding_size]
            elif typeIndex[0] == type2:
                # 表示Keyword Attention Distribution 模块 生成的单词概率最大
                # 此时index对应单词在keyword序列中的位置,注意keyword中的词可能是多个字符，因此需要特别处理一下
                # word是一个一维tensor,而不是一个标量
                word = keywords[i][Proba_keyword_index[i][0]]
                # 遍历word中每个字符，去除PAD位
                word = [id for id in word if id != PAD_ID]
                # 遍历去除填充符后的每个字符，将这些字符对应的概率设置为相等
                char_prob = torch.tensor(1/len(word))
                for id in word:
                    batch_word_prob_Distribution[i][id] = char_prob
                batch_word_Embedding[i] = keywordsEmbedding[i][Proba_keyword_index[i][0]].unsqueeze(0)  # [1,embedding_size]
        # 预测结果的概率分布，预测结果的Embedding
        return batch_word_prob_Distribution,batch_word_Embedding
    else:
        # 预测阶段
        # 当前batch中关键词的最大长度
        max_word_len = keywords.size(2)
        batch_word_ID = torch.zeros((batch_size, max_word_len))  # 每条预测结果中都存在大量的[PAD]符号(ID为0),在输出预测结果时需要去除
        # 当前解码结果对应的Embedding
        batch_word_Embedding = torch.zeros((batch_size, 1, embedding_size))
        # 预测阶段
        # 遍历每一条数据，获取其解码结果
        for i in range(batch_size):
            prob_list = torch.tensor([transformer_value[i][0], Proba_copy_value[i][0], Proba_keyword_value[i][0]])
            # 获取概率最大的模块,typeIndex: tensor([value])
            _, typeIndex = prob_list.topk(1)
            if typeIndex[0] == type0:
                # 表示TransformerDecoder生成模块生成的单词概率最大
                batch_word_ID[i][0] = transformer_index[i][0]  # 由于TransformerDecoder模块对应着词包,因此index就是单词ID
                wordEmbedding, _ = bertModel(transformer_index[i].unsqueeze(0))  # [1,1,embedding_size]
                batch_word_Embedding[i] = wordEmbedding[0]  # [1,embedding_size]
            elif typeIndex[0] == type1:
                # 表示Copy Attention Distribution 模块 生成的单词概率最大
                batch_word_ID[i][0] = Proba_copy_index[i][0]
                wordEmbedding, _ = bertModel(Proba_copy_index[i].unsqueeze(0))  # [1,1,embedding_size]
                batch_word_Embedding[i] = wordEmbedding[0]  # [1,embedding_size]
            elif typeIndex[0] == type2:
                # 表示Keyword Attention Distribution 模块 生成的单词概率最大
                # 此时index对应单词在keyword序列中的位置,注意keyword中的词可能是多个字符，因此需要特别处理一下
                # word是一个一维tensor,而不是一个标量
                word = keywords[i][Proba_keyword_index[i][0]]
                batch_word_ID[i] = word
                batch_word_Embedding[i] = keywordsEmbedding[i][Proba_keyword_index[i][0]].unsqueeze(0)  # [1,embedding_size]
        # 返回单词ID和单词Embedding
        return batch_word_ID,batch_word_Embedding