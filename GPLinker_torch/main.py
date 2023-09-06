# -*- coding: utf-8 -*-
"""
@Auth: Xhw
@Description: token-pair范式的实体关系抽取pytorch实现
"""
import torch
import json
import sys
import numpy as np
import torch.nn as nn
from nets.gpNet import RawGlobalPointer, sparse_multilabel_categorical_crossentropy
from transformers import BertTokenizerFast, BertModel
from utils.dataloader import data_generator, load_name ,search
from torch.utils.data import DataLoader
import configparser
# from torch.utils.tensorboard import SummaryWriter
from utils.bert_optimization import BertAdam
import utils.target_calculate
import os
os.environ["CUDA_VISIBLE_DEVICES"]='4'

con = configparser.ConfigParser()
con.read('./config.ini', encoding='utf8')
args_path = dict(dict(con.items('paths')), **dict(con.items("para")))
tokenizer = BertTokenizerFast.from_pretrained(args_path["model_path"], do_lower_case=True)
encoder = BertModel.from_pretrained(args_path["model_path"])

with open(args_path["schema_data"], 'r', encoding='utf-8') as f:
    schema = {}
    for idx, item in enumerate(f):
        item = json.loads(item.rstrip())
        schema[item["subject_type"]+"_"+item["predicate"]+"_"+item["object_type"]] = idx
id2schema = {}
for k,v in schema.items(): id2schema[v]=k

train_data = data_generator(load_name(args_path["train_file"]), tokenizer, max_len=con.getint("para", "maxlen"), schema=schema)
dev_data = data_generator(load_name(args_path["val_file"]), tokenizer, max_len=con.getint("para", "maxlen"), schema=schema)
train_loader = DataLoader(train_data , batch_size=con.getint("para", "batch_size"), shuffle=True, collate_fn=train_data.collate)
dev_loader = DataLoader(dev_data , batch_size=con.getint("para", "batch_size"), shuffle=True, collate_fn=dev_data.collate)

device = torch.device("cpu")

mention_detect = RawGlobalPointer(hiddensize=1024, ent_type_size=2, inner_dim=64).to(device)#实体关系抽取任务默认不提取实体类型
s_o_head = RawGlobalPointer(hiddensize=1024, ent_type_size=len(schema), inner_dim=64, RoPE=False, tril_mask=False).to(device)
s_o_tail = RawGlobalPointer(hiddensize=1024, ent_type_size=len(schema), inner_dim=64, RoPE=False, tril_mask=False).to(device)
class ERENet(nn.Module):
    def __init__(self, encoder, a, b, c):
        super(ERENet, self).__init__()
        self.encoder = encoder
        self.mention_detect = a
        self.s_o_head = b
        self.s_o_tail = c
        

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        #batch_token_ids [batch_size,几个句子中最长的那个句子的长度]有字为token，没字为0
        #batch_mask_ids [batch_size,几个句子中最长的那个句子的长度]有字为1，没字为0
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        #outputs 为bert模型输出(BaseModelOutputWithPoolingAndCrossAttentions)，分为两层
        #一个为outputs[0]，名称为last_hidden_state 宽度为[batchsize,maxlen,1024] 这里的1024为pretrain模型的最后一层dim
        #一个为outputs[1]，名称为pooler_output 宽度为[batchsize,1024] 这里的1024为pretrain模型的最后一层dim
        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        so_head_outputs = self.s_o_head(outputs, batch_mask_ids)
        so_tail_outputs = self.s_o_tail(outputs, batch_mask_ids)
        return mention_outputs, so_head_outputs, so_tail_outputs

net = ERENet(encoder, mention_detect, s_o_head, s_o_tail).to(device)
# optimizer = torch.optim.AdamW(
# 	net.parameters(),
#     lr=1e-5
# )
def set_optimizer(model, train_steps=None):
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=2e-5,
                         warmup=0.1,
                         t_total=train_steps)
    return optimizer

optimizer = set_optimizer(net, train_steps= (int(len(train_data) / con.getint("para", "batch_size")) + 1) * con.getint("para", "epochs"))
total_loss, total_f1 = 0., 0.

#测试集数据及其标签
with open(args_path["test_file"], encoding="utf-8") as f:
    test_text_list=[]
    test_true_spo=[]
    for text in f.readlines():
        test_text_list.append(json.loads(text.rstrip())["text"])
        test_true_spo.append(json.loads(text.rstrip())["spo_list"])

with open(args_path["train_file"], encoding="utf-8") as f:
    train_text_list=[]
    train_true_spo=[]
    for text in f.readlines():
        train_text_list.append(json.loads(text.rstrip())["text"])
        train_true_spo.append(json.loads(text.rstrip())["spo_list"])

train_text_list=train_text_list[0:50]
train_true_spo= train_true_spo[0:50]
test_text_list=test_text_list[0:50]
test_true_spo= test_true_spo[0:50]

test_info = {}
train_info = {}

test_info['ner_precision'] = []
test_info['ner_recall'] = []
test_info['ner_f1'] = []
test_info['rel_precision'] = []
test_info['rel_recall'] = []
test_info['rel_f1'] = []  
train_info['ner_precision'] = []
train_info['ner_recall'] = []
train_info['ner_f1'] = []
train_info['rel_precision'] = []
train_info['rel_recall'] = []
train_info['rel_f1'] = []  
train_info['loss'] = []  
# net.load_state_dict(torch.load('./erenet2.pth'))
best_ner_f1 = 0
best_rel_f1 = 0
for eo in range(con.getint("para", "epochs")):
    net.train()
    for idx, batch in enumerate(train_loader):
        text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = batch
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = \
            batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device), batch_head_labels.to(device), batch_tail_labels.to(device)
        logits1, logits2, logits3 = net(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss1 = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits1, mask_zero=True)
        loss2 = sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels, y_pred=logits2, mask_zero=True)
        loss3 = sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels, y_pred=logits3, mask_zero=True)
        loss = sum([loss1, loss2, loss3]) / 3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        sys.stdout.write("\r [EPOCH %d/%d] [Loss:%f] [idx:%f]"%(eo, con.getint("para", "epochs"), loss.item(),idx))
        train_info['loss'].append(loss.item())
    net.eval()
    print('\n')
    #train
    ner_precision,ner_recall,ner_f1,rel_precision,rel_recall,rel_f1=utils.target_calculate.model_test_printInfo(train_text_list,train_true_spo,tokenizer,net,device,id2schema)
    print('train ner|precision:',ner_precision,'|recall:',ner_recall,'|f1:',ner_f1)
    print('train rel|precision:',rel_precision,'|recall:',rel_recall,'|f1:',rel_f1)
    train_info['ner_precision'].append(ner_precision)
    train_info['ner_recall'].append(ner_recall)
    train_info['ner_f1'].append(ner_f1)
    train_info['rel_precision'].append(rel_precision)
    train_info['rel_recall'].append(rel_recall)
    train_info['rel_f1'].append(rel_f1)
    with open('logdata_train.json', 'w') as f:
        json.dump(train_info, f)
    #test
    ner_precision,ner_recall,ner_f1,rel_precision,rel_recall,rel_f1=utils.target_calculate.model_test_printInfo(test_text_list,test_true_spo,tokenizer,net,device,id2schema)
    print('test ner|precision:',ner_precision,'|recall:',ner_recall,'|f1:',ner_f1)
    print('test rel|precision:',rel_precision,'|recall:',rel_recall,'|f1:',rel_f1)
    test_info['ner_precision'].append(ner_precision)
    test_info['ner_recall'].append(ner_recall)
    test_info['ner_f1'].append(ner_f1)
    test_info['rel_precision'].append(rel_precision)
    test_info['rel_recall'].append(rel_recall)
    test_info['rel_f1'].append(rel_f1)
    with open('logdata_test.json', 'w') as f:
        json.dump(test_info, f)
    if ner_f1>best_ner_f1:
        best_ner_f1 = ner_f1
        torch.save(net.state_dict(), './erenet_ner_best.pth')
        print('save best ner model')
    if rel_f1>best_rel_f1:
        best_rel_f1 = rel_f1
        torch.save(net.state_dict(), './erenet_rel_best.pth')
        print('save best rel model')
    torch.save(net.state_dict(), './erenet.pth')



