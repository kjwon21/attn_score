import sys
sys.path.append('../custom_transformers')

op_mode = 0
if op_mode == 0:
    print("<<Normal mode>>\n")

    from my_transformers import Trainer, TrainingArguments
    from my_transformers import BertTokenizer, BertTokenizerFast, RobertaForQuestionAnswering, BertForQuestionAnswering, AdamW
    from my_transformers import ElectraTokenizer, ElectraTokenizerFast, ElectraForQuestionAnswering, AutoModelForQuestionAnswering
elif op_mode == 7:
    print("<<Random Token for shaking attention scores>>\n")

    from my_transformers21 import Trainer, TrainingArguments
    from my_transformers21 import BertTokenizer, BertTokenizerFast, RobertaForQuestionAnswering, BertForQuestionAnswering, AdamW
    from my_transformers21 import ElectraTokenizer, ElectraTokenizerFast, ElectraForQuestionAnswering, AutoModelForQuestionAnswering
elif op_mode == 8:
    print("<<Tied Positional Encoding>>\n")

    from my_transformers31 import Trainer, TrainingArguments
    from my_transformers31 import BertTokenizer, BertTokenizerFast, RobertaForQuestionAnswering, BertForQuestionAnswering, AdamW
    from my_transformers31 import ElectraTokenizer, ElectraTokenizerFast, ElectraForQuestionAnswering, AutoModelForQuestionAnswering
elif op_mode == 9:
    print("<<Adding embedding>>\n")

    from my_transformers41 import Trainer, TrainingArguments
    from my_transformers41 import BertTokenizer, BertTokenizerFast, RobertaForQuestionAnswering, BertForQuestionAnswering, AdamW
    from my_transformers41 import ElectraTokenizer, ElectraTokenizerFast, ElectraForQuestionAnswering, AutoModelForQuestionAnswering
else:
    raise Exception("Unknown mode!")
import transformers
transformers.logging.set_verbosity_error()

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
import json
import argparse
import time
import random, math
from konlpy.tag import Mecab          
from tqdm import tqdm

class KlueDataset(Dataset):
    def __init__(self, contexts, questions, answers, model_max_position_embedings, tokenizer, tokenizer2):
        self.tokenizer = tokenizer
        self.tokenizer2 = tokenizer2
        self.answers = answers
        self.questions = questions
        self.contexts = contexts
        self.model_max_position_embedings = model_max_position_embedings

        print("Tokenizing train_set... ", end='', flush=True)
        self.encodings = self.tokenizer(self.contexts, self.questions,
                                        max_length=512,
                                        truncation=True,
                                        padding="max_length",
                                        return_token_type_ids=False)
        self.encodings2 = self.tokenizer2(self.contexts, self.questions,
                                        max_length=512,
                                        truncation=True,
                                        padding="max_length",
                                        return_token_type_ids=False)
        print("  Done !!!")
        self.add_token_positions()
        
    def add_token_positions(self):
        start_positions = []
        end_positions = []
        for i in range(len(self.answers)):
            start_positions.append(self.encodings2.char_to_token(i, self.answers[i]['answer_start']))
            end_positions.append(self.encodings2.char_to_token(i, self.answers[i]['answer_end'] - 1))

            # positions 값이 None 값이라면, answer가 포함된 context가 잘렸다는 의미
            if start_positions[-1] is None:
                start_positions[-1] = self.model_max_position_embedings
            if end_positions[-1] is None:
                end_positions[-1] = self.model_max_position_embedings

        self.encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

    def get_data(self):
        return {"contexts":self.contexts, 'questions':self.questions, 'answers':self.answers}
    
    def get_encodings(self):
        return self.encodings
        
    def __getitem__(self, idx):
        return {key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings['input_ids'])

def DataProcess():
    def read_klue(path):
        with open(path, 'r', encoding='utf-8') as f:
            klue_dict = json.load(f)

        contexts = []
        questions = []
        answers = []
        for group in klue_dict['data']:
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    for answer in qa['answers']:
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)
        return contexts, questions, answers

    def read_dev_klue(path):
        with open(path, 'r', encoding='utf-8') as f:
            klue_dict = json.load(f)

        contexts = []
        questions = []
        answers = []
        for group in klue_dict['data']:
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    temp_answer = []
                    for answer in qa['answers']:
                        temp_answer.append(answer['text'])
                    if len(temp_answer) != 0: # answers의 길이가 0 == 답변할 수 없는 질문
                        contexts.append(context)
                        questions.append(question)
                        answers.append(temp_answer)
        return contexts, questions, answers

    def add_end_idx(answers, contexts):
        for answer, context in zip(answers, contexts):
            gold_text = answer['text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(gold_text)

            if context[start_idx:end_idx] == gold_text:
                answer['answer_end'] = end_idx
            elif context[start_idx-1:end_idx-1] == gold_text:
                answer['answer_start'] = start_idx - 1
                answer['answer_end'] = end_idx - 1
            elif context[start_idx-2:end_idx-2] == gold_text:
                answer['answer_start'] = start_idx - 2
                answer['answer_end'] = end_idx - 2

    if 'roberta' in model_path: 
        tokenizer = BertTokenizer.from_pretrained(tokenizer_roberta, prem=args.prem_prob, bf=args.bf_prob, model_input_names = ["input_ids", "attention_mask"])
        tokenizer_fast = BertTokenizerFast.from_pretrained(tokenizer_roberta, prem=args.prem_prob, bf=args.bf_prob, model_input_names = ["input_ids", "attention_mask"])
    elif 'electra' in model_path: 
        tokenizer = ElectraTokenizer.from_pretrained(tokenizer_electra, prem=args.prem_prob, bf=args.bf_prob)
        tokenizer_fast = ElectraTokenizerFast.from_pretrained(tokenizer_electra, prem=args.prem_prob, bf=args.bf_prob)
    else: 
        tokenizer = BertTokenizer.from_pretrained(tokenizer_bert, prem=args.prem_prob, bf=args.bf_prob, model_input_names = ["input_ids", "attention_mask"])
        tokenizer_fast = BertTokenizerFast.from_pretrained(tokenizer_bert, prem=args.prem_prob, bf=args.bf_prob)
    
    contexts, questions, answers = read_klue(data_folder + 'klue-mrc-v1.1_train.json')
    add_end_idx(answers, contexts)
    train_dataset = KlueDataset(contexts, questions, answers, 512, tokenizer, tokenizer_fast)

    d_contexts, d_questions, d_answers = read_dev_klue(data_folder + 'klue-mrc-v1.1_dev.json')
    e_contexts, t_contexts, e_questions, t_questions, e_answers, t_answers = train_test_split(d_contexts, d_questions, d_answers, test_size=.5, shuffle=True, random_state=split_seed)

    print("Tokenizing eval/test_set... ", end='', flush=True)
    eval_dataset = []
    for context, question in zip(e_contexts, e_questions):
        encodings = tokenizer(context, question, max_length=512, truncation=True,
                                padding="max_length", return_token_type_ids=False)
        encodings = {key: torch.tensor([val]) for key, val in encodings.items()}
        eval_dataset.append(encodings)

    test_dataset = []
    for context, question in zip(t_contexts, t_questions):
        encodings = tokenizer(context, question, max_length=512, truncation=True,
                                padding="max_length", return_token_type_ids=False)
        encodings = {key: torch.tensor([val]) for key, val in encodings.items()}
        test_dataset.append(encodings)
    print("  Done !!!")

    idx = random.randrange(0, len(train_dataset))
    print('ID %d :' %idx)
    print('C:', contexts[idx])
    print('Q:',  questions[idx])
    print(train_dataset[idx])
    print()
    print(f"Train set: {len(train_dataset)}, Eval set: {len(eval_dataset)}, Test set: {len(test_dataset)}")

    return tokenizer, train_dataset, eval_dataset, e_answers, test_dataset, t_answers

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

def main(tokenizer, train_dataset, eval_dataset, eval_answers, test_dataset, test_answers, bf=0.0):
    def train_runner(model, dataset, batch_size, num_train_epochs, learning_rate):       
        model.to(device)
        model.train()
        train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
        global_total_step = len(train_dataloader) * num_train_epochs
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0)

        with tqdm(total=global_total_step, unit='step') as t:
            total = 0
            total_loss = 0
            for epoch in range(num_train_epochs):
                for batch in train_dataloader:
                    optimizer.zero_grad()
                    input_ids = batch['input_ids'].to(device)
                    boost_mask = batch['boost_mask'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    outputs = model(input_ids,
                                boost_mask=boost_mask,
                                attention_mask=attention_mask,
                                start_positions=start_positions,
                                end_positions=end_positions)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item() * len(input_ids)
                    total += len(input_ids)
                    total_loss += batch_loss
                    global_total_step += 1
                    t.set_postfix(loss="{:.6f}".format(total_loss / total), batch_loss="{:.6f}".format(batch_loss))
                    t.update(1)
        model.save_pretrained(ckpt_path)
    
    def prediction(dataset):
        model.to(device)
        model.eval()
       
        result = []        
        with torch.no_grad():
            with tqdm(total=len(dataset)) as t:
                for encodings in dataset:                    
                    input_ids = encodings["input_ids"].to(device)
                    boost_mask = encodings['boost_mask'].to(device)
                    attention_mask = encodings["attention_mask"].to(device)
                    
                    outputs = model(input_ids, boost_mask=boost_mask, attention_mask=attention_mask)
                    start_logits, end_logits = outputs.start_logits, outputs.end_logits
                    token_start_index, token_end_index = start_logits.argmax(dim=-1), end_logits.argmax(dim=-1)
                    pred_ids = input_ids[0][token_start_index: token_end_index + 1]
                    pred = tokenizer.decode(pred_ids)
                    result.append(pred)
                    t.update(1)

        return result

    def em_evalutate(prediction_answers, real_answers):
        total = len(prediction_answers)
        exact_match = 0
        for prediction_answer, real_answer in zip(prediction_answers, real_answers):
            if prediction_answer in real_answer:
                exact_match += 1
        return (exact_match/total) * 100

    def model_load(path):
        if 'roberta' in model_path:  return RobertaForQuestionAnswering.from_pretrained(path)
        elif 'electra' in model_path:  return ElectraForQuestionAnswering.from_pretrained(path)
        else:  return BertForQuestionAnswering.from_pretrained(path)

    def model_init():
        if 'roberta' in model_path:  return RobertaForQuestionAnswering.from_pretrained(model_path)
        elif 'electra' in model_path:  return ElectraForQuestionAnswering.from_pretrained(model_path)
        else:  return BertForQuestionAnswering.from_pretrained(model_path)

    def model_reload():
        if 'roberta' in model_path:  return RobertaForQuestionAnswering.from_pretrained(ckpt_path)
        elif 'electra' in model_path:  return ElectraForQuestionAnswering.from_pretrained(ckpt_path)
        else:  return BertForQuestionAnswering.from_pretrained(ckpt_path)

    # model = model_load(model_path)
    # train_runner(model, train_dataset, train_batch, epoch_cnt, lr)

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
        eval_dataset=eval_dataset,        # evaluation dataset
    )
    trainer.train()

    change_bfactor(ckpt_path, bf)
    model = model_load(ckpt_path)

    pred_answers = prediction(eval_dataset)
    eval_em_score = em_evalutate(pred_answers, eval_answers)

    pred_answers = prediction(test_dataset)
    test_em_score = em_evalutate(pred_answers, test_answers)
    print("EM score (MRC) : %.2f,  (test: %.2f,)" % (eval_em_score, test_em_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="mrc_rt_mm.py")
    parser.add_argument("--start_bfactor", default=0.0, type=float, dest="start_bfactor")           #
    parser.add_argument("--end_bfactor", default=1.01, type=float, dest="end_bfactor")              #
    parser.add_argument("--inc_bfactor", default=0.05, type=float, dest="inc_bfactor")               #
    parser.add_argument("--use_prem_grp", default="no", type=str, dest="use_prem_grp")              #
    parser.add_argument("--use_lr_grp", default="no", type=str, dest="use_lr_grp")              
    parser.add_argument("--bf_prob", default=0.10, type=float, dest="bf_prob")              
    parser.add_argument("--prem_prob", default=0.0, type=float, dest="prem_prob")              
    parser.add_argument("--epoch_cnt", default=2, type=int, dest="epoch_cnt")              

    args = parser.parse_args()
    if op_mode != 7:
        args.bf_prob, args.prem_prob = 0.0, 0.0

    print("mrc_rt_mm.py - start_bfactor=%.2f, end_bfactor=%.2f, inc_bfactor=%.2f, use_prem_grp=%s, use_lr_grp=%s, bf_prob=%.2f, prem_prob=%.2f, epoch_cnt=%d" \
          % (args.start_bfactor, args.end_bfactor, args.inc_bfactor, args.use_prem_grp, args.use_lr_grp, args.bf_prob, args.prem_prob, args.epoch_cnt))

    tokenizer_electra = "../tokenizer/electra/"
    tokenizer_roberta = "../tokenizer/roberta/"
    tokenizer_bert = "../tokenizer/bert/"

    data_folder = "../dataset/klue-mrc-v1.1/"
    output_dir = 'ckpt/1/'
    model_list = [
                    './pretrained/klue-roberta-base_queans/',                               # 0
                    './pretrained/klue-roberta-small_queans/',                              # 1
                    './pretrained/klue-bert-base_queans/',                                  # 2
                    './pretrained/koelectra-base-v3-discriminator_queans/',                 # 3                 

                    './pretrained/roberta_pretrained/80k/rs_bf0.0/checkpoint-80000/',       # 4                 
                    './pretrained/roberta_pretrained/80k/rs_bf0.3/checkpoint-80000/',       # 5                 
                    './pretrained/roberta_pretrained/80k/rs_bf-0.3/checkpoint-80000/',      # 6                 
                    './pretrained/roberta_pretrained/80k/rs_bf0.2_prob0.08/checkpoint-80000/',      # 7                 
                    './pretrained/roberta_pretrained/80k/rs_pe/checkpoint-80000/',          # 8                 
                    './pretrained/roberta_pretrained/80k/rs_emb/checkpoint-80000/',         # 9                 
                ]

    start = time.time()                                             # 시작 시간 저장
    mid_list = [9]
    for mid in mid_list:
        model_path = model_list[mid]
        print("model : %s" % model_path)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        n_gpu = torch.cuda.device_count()
        print("number of gpus : %d" % n_gpu, flush=True)

        prem_grp = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        if args.use_prem_grp == "no":
            prem_grp = [1.0]

        # for epoch 4, mid=0, 1e-5: 60.23 / mid=1, 3e-5: 51.20 / mid=2, 3e-5: 55.39 / mid=3, 2e-5: 55.04
        lr_grp = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 10e-5]     
        if args.use_lr_grp == "no":
            lr_grp = [7e-5] #[1e-5, 1.5e-5]

        warmup_st, train_batch, split_seed = 0, 16, 2021
        epoch_cnt = args.epoch_cnt

        random.seed(split_seed)
        tokenizer, train_dataset, eval_dataset, eval_answers, test_dataset, test_answers = DataProcess()
        ckpt_path = output_dir + "checkpoint-" + str(epoch_cnt * math.ceil(len(train_dataset)/train_batch)) + "/"

        bf_muls = [(-1,0)]                   #  [(-1,0), (-1,1), (1,0), (1,1)] 
        for (bf_mul0, bf_mul1) in bf_muls:
            for lr in lr_grp:
                for prem in prem_grp:
                    print("<<< Learning rate: %f, bf_prob: %f, epoch_cnt: %d, split_seed: %d, BF_Mul: (%d, %d) >>>" % (lr, args.bf_prob, epoch_cnt, split_seed, bf_mul0, bf_mul1))
                    bf = args.start_bfactor
                    while bf <= args.end_bfactor:
                        change_bfactor(model_path, bf*bf_mul0)
                        main(tokenizer, train_dataset, eval_dataset, eval_answers, test_dataset, test_answers, bf*bf_mul1)
                        bf += args.inc_bfactor
                    print("\n\n")

    print("Elapsed time : %.1fs\n" %(time.time() - start))          # 현재시각 - 시작시간 = 실행 시간
