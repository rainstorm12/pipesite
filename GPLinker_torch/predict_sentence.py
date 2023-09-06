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
from GPLinker_torch.nets.gpNet import RawGlobalPointer, sparse_multilabel_categorical_crossentropy
from transformers import BertTokenizerFast, BertModel

# import configparser
# con = configparser.ConfigParser()
# con.read('./GPLinker_torch/config.ini', encoding='utf8')
# args_path = dict(dict(con.items('paths')), **dict(con.items("para")))
schema_path = './GPLinker_torch/datasets/standard/schemas.json'
premodel_path = 'GPLinker_torch/pretrain_model/RoBERTa_zh_Large_PyTorch'
tokenizer = BertTokenizerFast.from_pretrained(premodel_path, do_lower_case=True)
encoder = BertModel.from_pretrained(premodel_path)

with open(schema_path, 'r', encoding='utf-8') as f:
    schema = {}
    for idx, item in enumerate(f):
        item = json.loads(item.rstrip())
        schema[item["subject_type"]+"_"+item["predicate"]+"_"+item["object_type"]] = idx
id2schema = {}
for k,v in schema.items(): id2schema[v]=k

device = torch.device("cpu")
mention_detect = RawGlobalPointer(hiddensize=1024, ent_type_size=2, inner_dim=64).to(device)#实体关系抽取任务默认不提取实体类型
s_o_head = RawGlobalPointer(hiddensize=1024, ent_type_size=len(schema), inner_dim=64, RoPE=False, tril_mask=False).to(device)
s_o_tail = RawGlobalPointer(hiddensize=1024, ent_type_size=len(schema), inner_dim=64, RoPE=False, tril_mask=False).to(device)
class ERENet(nn.Module):
    def __init__(self, encoder, a, b, c):
        super(ERENet, self).__init__()
        self.mention_detect = a
        self.s_o_head = b
        self.s_o_tail = c
        self.encoder = encoder

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)

        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        so_head_outputs = self.s_o_head(outputs, batch_mask_ids)
        so_tail_outputs = self.s_o_tail(outputs, batch_mask_ids)
        return mention_outputs, so_head_outputs, so_tail_outputs

net = ERENet(encoder, mention_detect, s_o_head, s_o_tail).to(device)
net.load_state_dict(torch.load('./GPLinker_torch/erenet_rel_best.pth',map_location=torch.device('cpu')))
net.eval()

def model_test_GPLinker(text):
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=256)["offset_mapping"]
    new_span, entities = [], []
    for i in token2char_span_mapping:
        if i[0] == i[1]:
            new_span.append([])
        else:
            if i[0] + 1 == i[1]:
                new_span.append([i[0]])
            else:
                new_span.append([i[0], i[-1] - 1])
    threshold = 0.0
    encoder_txt = tokenizer.encode_plus(text, max_length=256)
    input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
    token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
    scores = net(input_ids, attention_mask, token_type_ids)
    outputs = [o[0].data.cpu().numpy() for o in scores]#output[0]为entity，[1]head,[2]tail
    subjects, objects = set(), set()
    #矩阵周围一圈全减去inf避免识别到头尾
    outputs[0][:, [0, -1]] -= np.inf#首尾减inf
    outputs[0][:, :, [0, -1]] -= np.inf#首尾减inf
    for l, h, t in zip(*np.where(outputs[0] > 0)):
        if l == 0:
            subjects.add((h, t))#（头实体）把大于0的位置加入到subject
        else:
            objects.add((h, t))#（尾实体）把大于0的位置加入到object
    spoes = set()
    #头实体节点再放入到head和tail中分别找出合适关系
    for sh, st in subjects:
        for oh, ot in objects:
            p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
            p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
            ps = set(p1s) & set(p2s)
            for p in ps:
                spoes.add((
                    text[new_span[sh][0]:new_span[st][-1] + 1], id2schema[p],
                    text[new_span[oh][0]:new_span[ot][-1] + 1]
                ))
    spo_list = []
    for spo in list(spoes):
        spo_list.append({"predicate":spo[1].split("_")[1], "object":spo[2], "object_type": spo[1].split("_")[2],
                            "subject":spo[0], "subject_type":spo[1].split("_")[0]
                            })
    return spo_list