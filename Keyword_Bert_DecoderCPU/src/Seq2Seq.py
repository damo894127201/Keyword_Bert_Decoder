# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 8:43
# @Author  : Weiyang
# @File    : Seq2Seq.py

#---------------------------------------------------------
# Seq2Seq+Attention
#---------------------------------------------------------

from BertEncoder import BertEncoder
from KeywordEncoder import KeywordEncoder
from Decoder import Decoder
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
from GreedySearch import GreedySearch

class Seq2Seq(nn.Module):

    def __init__(self,config,BertModel):
        super(Seq2Seq,self).__init__()
        self.output_size = config.output_size # 输出空间大小
        self.MAX_LENGTH = config.max_decoding_steps # 最大解码长度
        self.BertModel = BertModel # Bert模型
        self.embedding_size = config.embedding_size # 词向量的维度
        # 初始化encoder和decoder
        self.BertEncoder = BertEncoder(BertModel=BertModel)
        self.KeywordEncoder = KeywordEncoder(BertModel=BertModel,embedding_size=config.embedding_size)
        self.Decoder = Decoder(output_size=config.output_size,attention_size=config.attention_size,
                               embedding_size=config.embedding_size,feedforward_hidden_size=config.feedforward_hidden_size,
                               num_heads=config.num_heads,num_blocks=config.num_blocks,keep_prob=config.keep_prob)

    def forward(self,source,source_lens,target,keywords,keywords_len,teacher_forcing_ratio=0.5):
        '''
        :param source: [batch_size,max_time_step]
        :param source_lens: [batch_size]
        :param target: [batch_size,max_time_step]
        :param keywords: [batch_size,max_words_num,max_word_len],关键词序列
        :param keywords_len: [batch_size],每条数据真实的关键词个数
        :param teacher_forcing_ratio: 使用TeacherForcing训练的数据比例
        :return:outputs: [target_len,batch_size,vocab_size] # 各个时刻的解码结果，保持概率分布的形式
        '''
        target_len = target.size(1) # target的序列长度
        batch_size = target.size(0) # 当前批次大小
        max_time_step = source.size(1)  # 输入序列source的最大长度
        max_words_num = keywords.size(1) # 关键词序列中关键词的最大个数

        # Encoder source
        # 输入端各个时刻的字符的wordEmbeddings: [batch_size,max_time_step,embedding_size]
        # 表示输入端文本的句向量documentEmbedding: [batch_size,1,embedding_size]
        wordEmbeddings,documentEmbedding = self.BertEncoder(source)

        # Encoder keywords
        # [batch_size,max_words_num,embedding_size]
        keywordEmbeddings = self.KeywordEncoder(keywords)

        # 将输入序列source中的PAD位的wordEmbedding置为0,关键词序列中的PAD位的wordEmbedding置为0
        for i in range(batch_size):
            # source
            mask = torch.zeros((max_time_step,self.embedding_size),dtype=torch.float32) # [max_time_step,embedding_size]
            index = torch.arange(source_lens[i])
            mask[index] = torch.tensor(1,dtype=torch.float32)
            wordEmbeddings[i] = wordEmbeddings[i] * mask # [max_time_step,embedding_size]

            # keywords
            mask = torch.zeros((max_words_num,self.embedding_size),dtype=torch.float32)  # [max_words_num,embedding_size]
            index = torch.arange(keywords_len[i])
            mask[index] = torch.tensor(1, dtype=torch.float32)
            keywordEmbeddings[i] = keywordEmbeddings[i] * mask  # [max_time_step,embedding_size]

        # Encoder target
        # [batch_size,time_step,embedding_size]
        target,_ = self.BertEncoder(target)


        # Decoder
        # 以输出的各单元概率的形式，存储预测序列，便于交叉熵计算
        outputs = torch.zeros((target_len,batch_size,self.output_size))
        # 输入解码起始符_GO开始解码,需要给batch中每条数据都输入
        # 用[unused1]表示GO符号,[unused1]:1
        GO_token = torch.tensor([[1]]*batch_size) # [batch_size,1]
        GO_token,_ = self.BertModel(GO_token)
        # GO_token: [batch_size,1,embedding_size]
        # 将wordEmbeddings,documentEmbedding 与 GO_token拼接在一起，作为Decoder初始时刻的输入
        # [batch_size,(max_time_step-1)+1+1,embedding_size]
        decoder_input = torch.cat((wordEmbeddings,documentEmbedding,GO_token),1)

        # ---------------------------------------   Global Coverage  -----------------------------------------------
        # Coverage vector 覆盖向量，初始为0,用来累加解码过程各个时刻的注意力权重
        # CopyCoverageVector: [batch_size,max_time_step,1],max_time_step表示输入序列最大的长度
        # KeywordCoverageVector: [batch_size,max_words_num,1],max_words_num表示关键词序列中关键词的个数
        CopyCoverageVector = torch.zeros((batch_size,max_time_step,1))
        KeywordCoverageVector = torch.zeros((batch_size,max_words_num,1))
        # Coverage损失: CopyCoverageLoss+KeywordCoverageLoss
        GlobalCoverageLoss = torch.tensor(0,dtype=torch.float32)

        # 解码长度为 batch中的序列长度
        for step in range(target_len):
            decoderResult,CopyCoverageVector,KeywordCoverageVector,CoverageLoss = \
                self.Decoder(source,decoder_input,wordEmbeddings,documentEmbedding,keywordEmbeddings,
                             CopyCoverageVector,KeywordCoverageVector,flag=True)
            # 判断当前时刻解码结果的类型，即判断是TransformerDecoder模块、Copy Attention Distribution 模块
            # Keyword Attention Distribution 模块 哪个模块预测的结果，以方便输出对应的ID

            # batch_word_probDistribution: [batch_size,output_size]
            # batch_word_Embedding: [batch_size,1,embedding_size]
            batch_word_probDistribution, batch_word_Embedding = GreedySearch(decoderResult,self.output_size,self.BertModel,
                                                                   keywords,keywordEmbeddings,flag=True)
            outputs[step] = batch_word_probDistribution
            # 如果为True，则TeacherForcing
            TeacherForcing = random.random() < teacher_forcing_ratio # 随机生成[0,1)之间的数
            # target: [batch_size,max_time_step,embedding_size] ，取出下一时刻的一个batch的target
            next_target = target[:,step].unsqueeze(1) # [batch_size,1,embedding_size]
            next_decoder_input = next_target if TeacherForcing else batch_word_Embedding
            # 将当前解码的结果或下一时刻真实的target拼接到Decoder的输入中，作为下一时刻Decoder的输入
            decoder_input = torch.cat((decoder_input,next_decoder_input),1)
            GlobalCoverageLoss += CoverageLoss

        GlobalCoverageLoss = Variable(GlobalCoverageLoss, requires_grad=True)  # Coverage 损失

        return outputs,GlobalCoverageLoss # [max_time,batch_size,vocab_size],tensor标量

    def BatchSample(self,source,source_lens,keywords,keywords_len):
        '''
        批量预测
        :param source: batch输入，[batch,max_time_step]
        :param source_lens: [batch_size]
        :param keywords: [batch_size,max_words_num,max_word_len],关键词序列
        :param keywords_len: [batch_size],每条数据真实的关键词个数
        :return: 返回预测结果
        '''
        batch_size = source.size(0)  # 当前批次大小
        max_time_step = source.size(1)  # 输入序列source的最大长度
        max_words_num = keywords.size(1)  # 关键词序列中关键词的最大个数
        # Encoder source
        # 输入端各个时刻的字符的wordEmbeddings: [batch_size,max_time_step-1,embedding_size],去除了[CLS]的Embedding
        # 表示输入端文本的句向量: [batch_size,1,embedding_size]
        wordEmbeddings, documentEmbedding = self.BertEncoder(source)

        # Encoder keywords
        # [batch_size,max_words_num,embedding_size]
        keywordEmbeddings = self.KeywordEncoder(keywords)

        # 将输入序列source中的PAD位的wordEmbedding置为0,关键词序列中的PAD位的wordEmbedding置为0
        for i in range(batch_size):
            # source
            mask = torch.zeros((max_time_step, self.embedding_size),dtype=torch.float32)  # [max_time_step,embedding_size]
            index = torch.arange(source_lens[i])
            mask[index] = torch.tensor(1, dtype=torch.float32)
            wordEmbeddings[i] = wordEmbeddings[i] * mask  # [max_time_step,embedding_size]

            # keywords
            mask = torch.zeros((max_words_num, self.embedding_size),dtype=torch.float32)  # [max_words_num,embedding_size]
            index = torch.arange(keywords_len[i])
            mask[index] = torch.tensor(1, dtype=torch.float32)
            keywordEmbeddings[i] = keywordEmbeddings[i] * mask  # [max_time_step,embedding_size]

        # Decoder
        # 记录batch中，每条数据的各个时刻的预测结果,解码的最大长度为self.MAX_LENGTH
        results = []
        # 输入解码起始符_GO开始解码
        GO_token = torch.tensor([[1]]) # 用[unused1]表示GO，其在词典中索引为1
        GO_token, _ = self.BertModel(GO_token) # [1,1,embedding_size]
        # 解码终止符
        EOS_token = torch.tensor(2) # 用[unused2]表示EOS，其在词典中索引为2
        # 填充符
        PAD_token = torch.tensor(0) # [PAD]


        # 逐一对batch中的各条数据解码
        for i in range(batch_size):
            # 当前输入序列,[1,max_time_step]
            current_source = source[i].unsqueeze(0)
            # 存储当前序列的解码结果
            result = []
            # 当前序列bert编码后的结果
            wordEmbedding = wordEmbeddings[i].unsqueeze(0) # [1,max_time_step,embedding_size]
            document = documentEmbedding[i].unsqueeze(0) # [1,1,embedding_size]
            # 当前序列的关键词序列
            current_keyword = keywords[i].unsqueeze(0) # [1,max_words_num,max_word_len]
            # 当前序列关键词bert编码后的结果
            keywordEmbedding = keywordEmbeddings[i].unsqueeze(0) # [1,max_words_num,embedding_size]
            # 解码起始符
            decoder_input = torch.cat((wordEmbedding,document,GO_token),1) # [1,some_time,embedding_size]

            # ---------------------------------------   Global Coverage  -----------------------------------------------
            # Coverage vector 覆盖向量，初始为0,用来累加解码过程各个时刻的注意力权重
            # CopyCoverageVector: [1,max_time_step,1],max_time_step表示输入序列最大的长度
            # KeywordCoverageVector: [1,max_words_num,1],max_words_num表示关键词序列中关键词的个数
            CopyCoverageVector = torch.zeros((1, max_time_step, 1))
            KeywordCoverageVector = torch.zeros((1, max_words_num, 1))

            for j in range(self.MAX_LENGTH):
                decoderResult,CopyCoverageVector,KeywordCoverageVector,_ = \
                    self.Decoder(current_source,decoder_input,wordEmbedding,document,keywordEmbedding,
                                 CopyCoverageVector,KeywordCoverageVector,flag=False)

                # 判断当前时刻解码结果的类型，即判断是TransformerDecoder模块、Copy Attention Distribution 模块
                # Keyword Attention Distribution 模块 哪个模块预测的结果，以方便输出对应的ID

                # batch_word_ID: [1,max_word_len]
                # batch_word_Embedding: [1,1,embedding_size]
                batch_word_ID,batch_word_Embedding = GreedySearch(decoderResult,self.output_size,self.BertModel,
                                                                                 current_keyword, keywordEmbedding, flag=False)
                # 将当前解码的结果拼接到Decoder的输入中，作为下一时刻Decoder的输入
                decoder_input = torch.cat((decoder_input,batch_word_Embedding),1)

                # 去除batch_word_ID中的PAD
                wordID = torch.tensor([id for id in batch_word_ID[0] if id != PAD_token])
                result.extend(wordID.tolist())
                if len(wordID) == 1:
                    if wordID[0] == EOS_token:
                        break
            results.append(result)
        return results # [batch_size,some_time_step]

if __name__ == '__main__':
    from Config import Config
    config = Config()
    model = Seq2Seq(config)
    print(model)