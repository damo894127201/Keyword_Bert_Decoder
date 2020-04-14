# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 8:43
# @Author  : Weiyang
# @File    : MyDataSet.py

#----------------------------------------------------
# 自定义数据集，继承torch.utils.data.Dataset
# 这里的tokenizer直接使用Transformers库：transformers.tokenization_bert.BertTokenizer
# [PAD] ,解码起始符_GO用[unused1]表示,解码终止符_EOS用[unused2]表示,[UNK] 表示未知字符
#----------------------------------------------------

from torch.utils.data import Dataset

class MyDataSet(Dataset):

    def __init__(self,source_path,target_path,keyword_path,tokenizer):
        self.tokenizer = tokenizer
        self.sourceData,self.targetData ,self.keywordData = self.load(source_path,target_path,keyword_path)

    def load(self,source_path,target_path,keyword_path):
        '''load source data'''
        with open(source_path, 'r', encoding='utf-8') as fs, open(target_path, 'r', encoding='utf-8') as ft,open(keyword_path,'r',encoding='utf-8') as fk:
            sourceData = []
            targetData = []
            keywordData = []
            EOS_id = self.tokenizer.convert_tokens_to_ids(['[unused2]'])[0] # 用[unused2]表示EOS符号
            CLS_id = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0] # 文章起始符
            SEP_id = self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0] # 句子分隔符
            for source,target,keyword in zip(fs, ft,fk):
                sources = source.strip().split('\t') # 切割句子
                temp = [CLS_id] # 存储当前文章所有字符对应的ID，需在文章起始处添加[CLS]符
                for i,sentence in enumerate(sources):
                    ids = self.tokenizer.encode(sentence,add_special_tokens=False)# tokenizer不在句前和句尾加[CLS]和[SEP]符号
                    temp.extend(ids) # 当前句子ID序列
                    temp.append(SEP_id) # 句子尾部添加[SEP]
                sourceData.append(temp)

                # target 末尾加上EOS结束符,且 tokenizer不在句前和句尾加[CLS]和[SEP]符号
                targetData.append(self.tokenizer.encode(target.strip(),add_special_tokens=False)+[EOS_id])

                # 由于关键词是多个字符构成的单词，而BERT是基于字符的，而我们最终需要获得的是这个单词经过BERT处理后的向量表示
                # 因此，我们不能把关键词序列处理成一个个字符的ID序列，需要将同一个关键词的ID放在一个列表，故而，一篇文本
                # 的关键词序列是一个二维的列表[[id,id],[id,id],..]
                words = keyword.strip().split()
                temp = []
                for word in words:
                    ids = self.tokenizer.encode(' '.join(word),add_special_tokens=False)
                    temp.append(ids)
                keywordData.append(temp)

        return sourceData,targetData,keywordData

    def __len__(self):
        '''要求：返回数据集大小'''
        return len(self.sourceData)

    def __getitem__(self,index):
        '''要求：传入index后，可按index单例或切片返回'''
        return self.sourceData[index],self.targetData[index],self.keywordData[index]