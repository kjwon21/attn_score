import sys
sys.path.append('../custom_transformers')

op_mode = 0
if op_mode == 0:
    print("<<Normal mode>>\n")

    from my_transformers import Trainer, TrainingArguments
    from my_transformers import BertTokenizer, RobertaForSequenceClassification, BertForSequenceClassification
    from my_transformers import ElectraTokenizer, ElectraForSequenceClassification
elif op_mode == 7:
    print("<<Random Token for shaking attention scores>>\n")

    from my_transformers21 import Trainer, TrainingArguments
    from my_transformers21 import BertTokenizer, RobertaForSequenceClassification, BertForSequenceClassification
    from my_transformers21 import ElectraForPreTraining, ElectraTokenizer, ElectraForSequenceClassification
elif op_mode == 8:
    print("<<Tied Positional Encoding>>\n")

    from my_transformers31 import Trainer, TrainingArguments
    from my_transformers31 import BertTokenizer, RobertaForSequenceClassification, BertForSequenceClassification
    from my_transformers31 import ElectraForPreTraining, ElectraTokenizer, ElectraForSequenceClassification
elif op_mode == 9:
    print("<<Adding embedding>>\n")

    from my_transformers41 import Trainer, TrainingArguments
    from my_transformers41 import BertTokenizer, RobertaForSequenceClassification, BertForSequenceClassification
    from my_transformers41 import ElectraForPreTraining, ElectraTokenizer, ElectraForSequenceClassification
else:
    raise Exception("Unknown mode!")

import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
import json
import argparse
import time
import random
import math
from konlpy.tag import Mecab          

class TDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def DataProcess():
    if 'roberta' in model_path: tokenizer = BertTokenizer.from_pretrained(tokenizer_roberta, prem=args.prem_prob, bf=args.bf_prob, model_input_names = ["input_ids", "attention_mask"])
    elif 'electra' in model_path: tokenizer = ElectraTokenizer.from_pretrained(tokenizer_electra, prem=args.prem_prob, bf=args.bf_prob)
    else: tokenizer = BertTokenizer.from_pretrained(tokenizer_bert, prem=args.prem_prob, bf=args.bf_prob, model_input_names = ["input_ids", "attention_mask"])
    
    df = pd.read_csv(data_folder + "SKT_BoolQ_Train.tsv", delimiter='\t')
    df.drop(['ID'], axis=1, inplace=True)
    df.columns = ['sentence1', 'sentence2', 'label']

    df2 = pd.read_csv(data_folder + "SKT_BoolQ_Dev.tsv", delimiter='\t')
    df2.drop(['ID'], axis=1, inplace=True)
    df2.columns = ['sentence1', 'sentence2', 'label']

    sep = " [SEP] "
    train_texts = []
    for _, row in df.iterrows():
        train_texts.append(row['sentence2'] + sep + row['sentence1'])
    train_labels = df['label'].values

    test_texts = []
    for _, row in df2.iterrows():
        test_texts.append(row['sentence2'] + sep + row['sentence1'])
    test_labels = df2['label'].values

    val_texts_t, test_texts_t, val_labels_t, test_labels_t = train_test_split(test_texts, test_labels, test_size=.5, shuffle=True, random_state=split_seed)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    train_dataset = TDataset(train_encodings, train_labels)

    val_encodings = tokenizer(val_texts_t, truncation=True, padding=True)
    val_dataset = TDataset(val_encodings, val_labels_t)     
    tensor_val_labels = [val_dataset[i]['labels'] for i in range(len(val_dataset))]

    test_encodings = tokenizer(test_texts_t, truncation=True, padding=True)
    test_dataset = TDataset(test_encodings, test_labels_t)
    tensor_test_labels = [test_dataset[i]['labels'] for i in range(len(test_dataset))]

    k_tokenizer = Mecab()
    print('\ntrain_set : %d, val_set: %d, test_set: %d' %(len(train_dataset), len(val_dataset), len(test_dataset)))
    idx = random.randrange(0, len(train_texts))
    tlist = tokenizer.tokenize(train_texts[idx])
    idlist = tokenizer.convert_tokens_to_ids(tlist)
    k_tokens = k_tokenizer.pos(train_texts[idx])

    print('ID %d - ' %idx, end = '')
    print(train_texts[idx])
    print(k_tokens)
    print()
    print(tlist)
    print(idlist)
    print(train_dataset[idx])
    print()

    return train_dataset, val_dataset, tensor_val_labels, test_dataset, tensor_test_labels

def change_bfactor(path, bf=0.0, b_elayer=0):
    j_file = path + "config.json"
    with open(j_file, 'r', encoding='utf-8') as jf:
        json_data = json.load(jf)

    json_data["boost_factor"] = bf
    json_data["boost_prem_grp"] = prem
    json_data["boost_elayer"] = b_elayer
    print("\nboost_factor : %.2f, boost_prem_grp : %.2f" % (bf, prem))

    with open(j_file, 'w', encoding='utf-8') as jf:
        json.dump(json_data, jf, indent=4)

def main(train_dataset, val_dataset, val_labels, test_dataset, test_labels, bf=0.0):
    def model_init():
        if 'roberta' in model_path:  return RobertaForSequenceClassification.from_pretrained(model_path)
        elif 'electra' in model_path:  return ElectraForSequenceClassification.from_pretrained(model_path)
        else:  return BertForSequenceClassification.from_pretrained(model_path)

    def model_reload():
        if 'roberta' in model_path:  return RobertaForSequenceClassification.from_pretrained(ckpt_path)
        elif 'electra' in model_path:  return ElectraForSequenceClassification.from_pretrained(ckpt_path)
        else:  return BertForSequenceClassification.from_pretrained(ckpt_path)

    def predict(trainer, dataset, labels):
        predictions = trainer.predict(test_dataset = dataset)
        predictions = predictions[0]
        predictions = torch.tensor(predictions.tolist())
        res_labels = torch.argmax(predictions, dim=1)

        acc = sum(x == y for x, y in zip(labels, res_labels)) / len(labels) * 100
        return acc.item()

    training_args = TrainingArguments(
        output_dir=output_dir,           # output directory
        num_train_epochs=epoch_cnt,      # total number of training epochs
        per_device_train_batch_size=train_batch,
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=warmup_st,          # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./log',             # directory for storing logs
        logging_strategy="epoch",
        # logging_steps=1000,
        evaluation_strategy="no",        # "no", "epoch"
        save_strategy="epoch",           ##
        load_best_model_at_end=False,     ##
        save_total_limit=epoch_cnt,
        learning_rate=lr, 
        seed=2022,
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,              # training arguments, defined above
        train_dataset=train_dataset,     # training dataset
        eval_dataset=val_dataset,        # evaluation dataset
    )
    trainer.train()

    change_bfactor(ckpt_path, bf)
    trainer = Trainer(
        model_init=model_reload,
        args=training_args,              # training arguments, defined above
    )

    e_acc = predict(trainer, val_dataset, val_labels)
    t_acc = predict(trainer, test_dataset, test_labels)
    print("Accuracy (BoolQ) : %.2f,  (test: %.2f,)" % (e_acc, t_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="boolq_r_mm.py")
    parser.add_argument("--start_bfactor", default=0.0, type=float, dest="start_bfactor")           #
    parser.add_argument("--end_bfactor", default=1.01, type=float, dest="end_bfactor")              #
    parser.add_argument("--inc_bfactor", default=0.1, type=float, dest="inc_bfactor")               #
    parser.add_argument("--use_prem_grp", default="no", type=str, dest="use_prem_grp")              #
    parser.add_argument("--use_lr_grp", default="no", type=str, dest="use_lr_grp")              
    parser.add_argument("--bf_prob", default=0.10, type=float, dest="bf_prob")              
    parser.add_argument("--prem_prob", default=0.0, type=float, dest="prem_prob")              
    parser.add_argument("--epoch_cnt", default=2, type=int, dest="epoch_cnt")              

    args = parser.parse_args()
    if op_mode != 7:
        args.bf_prob, args.prem_prob = 0.0, 0.0

    print("boolq_r_mm.py - start_bfactor=%.2f, end_bfactor=%.2f, inc_bfactor=%.2f, use_prem_grp=%s, use_lr_grp=%s, bf_prob=%.2f, prem_prob=%.2f, epoch_cnt=%d" \
          % (args.start_bfactor, args.end_bfactor, args.inc_bfactor, args.use_prem_grp, args.use_lr_grp, args.bf_prob, args.prem_prob, args.epoch_cnt))

    tokenizer_electra = "../tokenizer/electra/"
    tokenizer_roberta = "../tokenizer/roberta/"
    tokenizer_bert = "../tokenizer/bert/"

    data_folder = "../dataset/boolq/"
    output_dir = 'ckpt/1/'
    model_list = [
                    './pretrained/klue-roberta-base_seqcls/',                               # 0
                    './pretrained/klue-roberta-small_seqcls/',                              # 1
                    './pretrained/klue-bert-base_seqcls/',                                  # 2
                    './pretrained/koelectra-base-v3-discriminator_seqcls/',                 # 3                 

                    './pretrained/roberta_pretrained/80k/rs_bf0.0/checkpoint-80000/',       # 4                 
                    './pretrained/roberta_pretrained/80k/rs_bf0.3/checkpoint-80000/',       # 5                 
                    './pretrained/roberta_pretrained/80k/rs_bf-0.3/checkpoint-80000/',      # 6                 
                    './pretrained/roberta_pretrained/80k/rs_bf0.2_prob0.08/checkpoint-80000/',      # 7                 
                    './pretrained/roberta_pretrained/80k/rs_pe/checkpoint-80000/',          # 8                 
                    './pretrained/roberta_pretrained/80k/rs_emb/checkpoint-80000/',         # 9                 

                    './pretrained/roberta_pretrained/60k/rs_bf0.0/checkpoint-60000/',       # 10                 
                    './pretrained/roberta_pretrained/60k/rs_bf0.3/checkpoint-60000/',       # 11                
                    './pretrained/roberta_pretrained/60k/rs_bf-0.3/checkpoint-60000/',      # 12                
                    './pretrained/roberta_pretrained/60k/rs_bf0.2_prob0.08/checkpoint-60000/',      # 13                 
                    './pretrained/roberta_pretrained/60k/rs_pe/checkpoint-60000/',          # 14                 
                    './pretrained/roberta_pretrained/60k/rs_emb/checkpoint-59800/',         # 15                
                ]
    
    start = time.time()                                             # 시작 시간 저장
    mid_list = [11]
    for mid in mid_list:
        model_path = model_list[mid]
        print("model : %s" % model_path)

        n_gpu = torch.cuda.device_count()
        print("number of gpus : %d" % n_gpu, flush=True)

        prem_grp = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        if args.use_prem_grp == "no":
            prem_grp = [1.0]

        lr_grp = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 10e-5]
        if args.use_lr_grp == "no":
            lr_grp = [8e-5]            # [3.5e-5, 4.5e-5, 5e-5]

        warmup_st, train_batch, split_seed = 0, 32, 2022
        epoch_cnt = args.epoch_cnt
        
        train_dataset, val_dataset, val_labels, test_dataset, test_labels = DataProcess()
        ckpt_path = output_dir + "checkpoint-" + str(epoch_cnt * math.ceil(len(train_dataset)/train_batch)) + "/"

        bf_muls = [(1,0)]                  #  [(-1,0), (-1,1), (1,0), (1,1)] 
        for (bf_mul0, bf_mul1) in bf_muls:
            for lr in lr_grp:
                for prem in prem_grp:
                    print("<<< Learning rate: %f, bf_prob: %f, epoch_cnt: %d, split_seed: %d, BF_Mul: (%d, %d) >>>" % (lr, args.bf_prob, epoch_cnt, split_seed, bf_mul0, bf_mul1))
                    bf = args.start_bfactor
                    while bf <= args.end_bfactor:
                        change_bfactor(model_path, bf*bf_mul0)
                        main(train_dataset, val_dataset, val_labels, test_dataset, test_labels, bf*bf_mul1)
                        bf += args.inc_bfactor
                    print("\n\n")

    print("Elapsed time : %.1fs\n" %(time.time() - start))          # 현재시각 - 시작시간 = 실행 시간
