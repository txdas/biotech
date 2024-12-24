import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torch.nn.functional import pad
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
model_name ="bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
DATADIR = "C:\\Users\\jinya\\Desktop\\202501\\2025_1学期\\data\\nlp\\大作业数据集\\中文地址相关性"
labels = ["not_match","partial_match","exact_match"]
label2id = { v:i  for i, v in enumerate(labels)}
id2label = {i:v for i,v in enumerate(labels)}



class DFData(Dataset):
    def __init__(self,src):
        self.src = src

    def __getitem__(self, item):
        value = self.src.iloc[item]
        text = value["sentence1"]+tokenizer.sep_token+value["sentence2"]
        encoded = tokenizer(text, return_tensors='pt')
        encoded["labels"] = [label2id[value["label"]]]
        return encoded

    def __len__(self):
        return len(self.src)



train_data= pd.read_json(f"{DATADIR}\\train.json",orient="records",lines=True)
eval_data= pd.read_json(f"{DATADIR}\\test.json",orient="records",lines=True)
train_dataset, eval_dataset = DFData(train_data),DFData(eval_data)
train_dataset[0]

import torch
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer


def collate_fn(batch):
    datas = [v for v in batch]
    print(len(datas),datas[0])
    max_len = max([ len(v["input_ids"][0]) for v in datas])
    for v in datas:
        pad_len = max_len - (len(v["input_ids"][0]))
        if pad_len>0:
            v["input_ids"] = pad(v["input_ids"],(0,pad_len),value=tokenizer.pad_token_type_id)
            v["attention_mask"] = pad(v["attention_mask"],(0,pad_len),value=0)
            v["token_type_ids"] = pad(v["token_type_ids"],(0,pad_len),value=0)
    print(torch.tensor([v["labels"] for v in datas]).shape)
    return {"input_ids": torch.cat([v["input_ids"]for v in datas],dim=0),
            "attention_mask": torch.cat([v["attention_mask"] for v in datas],dim=0),
            "token_type_ids":torch.cat([v["token_type_ids"] for v in datas],dim=0),
            "labels": torch.tensor([v["labels"] for v in datas])}

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc,"f1": f1,"precision": precision,"recall": recall }

model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(labels))
train_args = TrainingArguments(logging_steps=25, per_device_train_batch_size=16,
                               eval_steps= 25, num_train_epochs=1, output_dir="similar")
trainer = Trainer(model=model,args=train_args, train_dataset=train_dataset,
                  eval_dataset=eval_dataset, data_collator=collate_fn, compute_metrics=compute_metrics)
trainer.train()
model.save_pretrained("similar/models", from_pt=True)