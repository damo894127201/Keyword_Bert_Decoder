# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 8:43
# @Author  : Weiyang
# @File    : TrainModel.py

#-----------------------------------------------
# 断点训练模型
#-----------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from MyDataSet import MyDataSet
from transformers.modeling_bert import BertModel,BertConfig
from transformers.tokenization_bert import BertTokenizer
from pad import pad
from sequence_mask_loss import sequence_mask_loss
from Config import Config
from Seq2Seq import Seq2Seq
from rouge import Rouge
from convert_to_RougePattern import convert_to_RougePattern
import random

if __name__ == '__main__':
    source_path = '../data/train/source.txt'
    target_path = '../data/train/target.txt'
    keyword_path = '../data/train/TextRank.txt'
    eval_source_path = '../data/eval/source.txt'
    eval_target_path = '../data/eval/target.txt'
    eval_keyword_path = '../data/eval/TextRank.txt'
    log_path = '../log/log.txt'
    bert_path = '../../../chinese_wwm_ext_pytorch/'
    pre_trainModel = '../model/model.pth7000' # 断点训练的模型名称
    log = open(log_path,'w',encoding='utf-8')
    rouge = Rouge()  # 评估指标

    # 加载BERT模型
    bert_config = BertConfig.from_pretrained(bert_path + 'bert_config.json') # 配置文件
    bert_model = BertModel.from_pretrained(bert_path + 'pytorch_model.bin', config=bert_config) # 模型
    tokenizer = BertTokenizer.from_pretrained(bert_path + 'vocab.txt') # 词包
    config = Config()

    # 训练集
    loader = DataLoader(dataset=MyDataSet(source_path, target_path,keyword_path, tokenizer), batch_size=config.batch_size, shuffle=True,
                        num_workers=2,collate_fn=pad,drop_last=False) # 最后一个batch数据集不丢弃
    # 评估集
    eval_loader = DataLoader(dataset=MyDataSet(eval_source_path, eval_target_path,eval_keyword_path, tokenizer),batch_size=config.batch_size,
                             shuffle=True, num_workers=2, collate_fn=pad, drop_last=False)  # 最后一个batch数据集不丢弃

    model = Seq2Seq(config,bert_model)
    checkpoint = torch.load(pre_trainModel)
    model.load_state_dict(checkpoint['model'])
    #optimizer = optim.SGD(model.parameters(),lr=config.learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = nn.CrossEntropyLoss()

    total_loss_criterion = 0
    PAD_ID = 0
    print_every = 20000
    num_training_steps = 600000 # 至少训练多少个step
    print_every_loss = [] # 记录每print_every个step的batch大小数据的平均损失，用于输出查看损失曲线
    num_eval = 8  # 评估集最少评估次数
    Rouge_threshold = 3  # 评估集上前后num_eval次评估时，ROUGE差值若低于该值，则满足早停条件，ROUGE采用百分制
    RougeL = []  # 存储每次在评估集上得到的ROUGEL平均指标

    step = 1
    startEpoch = checkpoint['Epoch']
    for epoch in range(startEpoch,config.num_Epochs):
        for iter,(batch_x,batch_y,batch_source_lens,batch_target_lens,batch_keyword,batch_keyword_lens) in enumerate(loader):
            # batch_x,batch_y:[ batch_size,max_time_step]; batch_source_lens,batch_target_lens:[batch_size,]
            # batch_keyword:[batch_size,max_words_num,max_word_len]
            '''
            print('Epoch: ',epoch,'| step: ',step,'|batch_x: ',batch_x,'|batch_y: ',batch_y,'|batch_lens: ',batch_source_lens,
                  '|batch_keyword: ',batch_keyword,'|batch_keyword_lens: ',batch_keyword_lens)
            print(batch_x.size())
            print(batch_y.size())
            print(batch_keyword.size())
            print(batch_keyword_lens)
            exit()
            '''
            optimizer.zero_grad() # 每批次训练时初始梯度得置0
            batch_loss = 0 # 每批次数据的损失
            batch_size = batch_x.size(0) # 每batch的大小
            target_max_length = batch_y.size(1)
            # outputs: [max_time_step,batch_size,vocab_size]
            outputs,GlobalCoverageLoss = model(batch_x,batch_source_lens,batch_y,batch_keyword,batch_keyword_lens,config.TeacherForcingRate)
            # 处理PAD填充位，避免对其进行损失计算
            outputs = sequence_mask_loss(outputs,batch_target_lens,config.output_size)
            # 计算所条样例的损失,需要将每一时刻所有样例的的交叉熵损失累积
            batch_y = batch_y.view(-1,batch_size) # [max_time_step,batch_size]
            for j in range(target_max_length):
                # 真实label计算时会自动转为one-hot形式
                batch_loss += criterion(outputs[j], batch_y[j])

            batch_loss = batch_loss + config.ratio * GlobalCoverageLoss
            batch_loss = Variable(batch_loss, requires_grad=True)
            batch_loss.backward() # 反向传播梯度更新
            nn.utils.clip_grad_norm_(model.parameters(),config.max_norm,config.norm_type) # 梯度裁剪
            optimizer.step()
            total_loss_criterion += float(batch_loss)

            if step % 200 == 0:
                average_loss = total_loss_criterion / step
                print('Epoch: %3d' % (epoch + 1), '\t| Step: %5d' % (step), '\t| Average_loss: %10.2f' % (average_loss),
                      '\t| Total_loss: %10.2f' % (total_loss_criterion),flush=True)

            if step % print_every == 0:
                average_loss = total_loss_criterion / step
                print_every_loss.append(round(average_loss,2))
                # 保存模型参数，如果也保存优化器参数，便可以在此基础上接断点继续训练
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'Epoch': epoch}
                torch.save(state, '../model/model.pth' + str(step))

                # 评估
                batch_eval_rouge1 = []
                batch_eval_rouge2 = []
                batch_eval_rougeL = []
                for eval_iter, (batch_eval_x,batch_eval_y,batch_eval_source_lens,batch_eval_target_lens,batch_eval_keyword,batch_eval_keyword_lens) in enumerate(eval_loader):
                    eval_outputs = model.BatchSample(batch_eval_x,batch_eval_source_lens,batch_eval_keyword,batch_eval_keyword_lens)
                    # 随机输出一条预测结果
                    '''
                    batch_eval_size = batch_eval_y.size(0)
                    sample_index = random.randint(0,batch_eval_size-1)
                    true_words = tokenizer.convert_ids_to_tokens(batch_eval_y[sample_index].tolist())
                    predict_words = tokenizer.convert_ids_to_tokens(eval_outputs[sample_index])
                    print('True: ' + ''.join(true_words))
                    print('Predict: ' + ''.join(predict_words))
                    print()
                    '''
                    # 批量评估
                    # eval_outputs转换后的格式: ['id id id id','id id id',....],需要去除填充位PAD(id=0)和终止符_EOS(id=2)
                    # batch_eval_y转换后的格式: ['id id id id','id id id',....],需要去除填充位PAD(id=0)和终止符_EOS(id=2)
                    eval_outputs, batch_eval_y = convert_to_RougePattern(eval_outputs, batch_eval_y)
                    rouge_score = rouge.get_scores(eval_outputs, batch_eval_y)
                    # 获取ROUGE-1、ROUGE-2、ROUGE-L
                    for i in range(len(eval_outputs)):
                        batch_eval_rouge1.append(rouge_score[i]['rouge-1']['r'])
                        batch_eval_rouge2.append(rouge_score[i]['rouge-2']['r'])
                        batch_eval_rougeL.append(rouge_score[i]['rouge-l']['r'])
                # 计算ROUGE各指标的平均值
                num_data = len(batch_eval_rouge1)
                batch_eval_rouge1 = sum(batch_eval_rouge1) * 100 / num_data
                batch_eval_rouge2 = sum(batch_eval_rouge2) * 100 / num_data
                batch_eval_rougeL = sum(batch_eval_rougeL) * 100 / num_data
                # 输出当前step,评估集的ROUGE指标
                line = 'Epoch: %3d' % (epoch + 1) + '\t| Step: %5d' % step + '\t| ROUGE-1: %10.2f' % batch_eval_rouge1 \
                       + '\t| ROUGE-2: %10.2f' % batch_eval_rouge2 + '\t| ROUGE-L: %10.2f' % batch_eval_rougeL
                print(line,flush=True)
                log.write(line)
                log.write('\n')

                # 存储当前ROUGE平均值
                RougeL.append(batch_eval_rougeL)

            if step > num_training_steps and len(RougeL) > num_eval:
                # 判断最近num_eval次在评估集上的ROUGE结果，是否收敛
                # ROUGEL
                Recent_rougeL_1 = torch.tensor(RougeL[-num_eval:])  # 最近num_eval次评估集上的ROUGEL指标
                Recent_rougeL_2 = torch.tensor(RougeL[-(num_eval + 1):-1])
                threshold = Recent_rougeL_1 - Recent_rougeL_2
                thresholdL = torch.sum(torch.abs(threshold)) / num_eval

                if thresholdL < Rouge_threshold:
                    # 保存模型参数，如果也保存优化器参数，便可以在此基础上接断点继续训练
                    print('-'*20,'Training stopped at ','\tEpoch:',epoch+1,'\t| Step: ',step,'!','-'*20)
                    log.write('print_every_loss:\t')
                    log.write('\t'.join([str(element) for element in print_every_loss]))
                    log.close()
                    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'Epoch': epoch}
                    torch.save(state, '../model/model.pth')
                    exit()
            step += 1
        total_loss_criterion = 0

    # 保存模型参数，如果也保存优化器参数，便可以在此基础上接断点继续训练
    state = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'Epoch':epoch}
    torch.save(state,'../model/model.pth')
    log.write('print_every_loss:\t')
    log.write('\t'.join([str(element) for element in print_every_loss]))
    log.close()