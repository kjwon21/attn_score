import os, sys
import tqdm
import torch
from pathlib import Path
import json
import argparse
import time
import random
import functools

sys.path.append('../roberta/custom_transformers')
if True:       # boost_mask = [(), ()]
    from my_transformers2 import RobertaConfig
    from my_transformers2 import RobertaForMaskedLM
    from my_transformers2 import BertTokenizer, RobertaTokenizerFast
    from my_transformers2 import DataCollatorForLanguageModeling
    from my_transformers2 import Trainer, TrainingArguments
else:           # same as previous test. boost_mask = [ , , ,]
    from my_transformers import RobertaConfig
    from my_transformers import RobertaForMaskedLM
    from my_transformers import BertTokenizer, RobertaTokenizerFast
    from my_transformers import DataCollatorForLanguageModeling
    from my_transformers import Trainer, TrainingArguments

# from nlp import load_dataset
from datasets import load_dataset

from transformers.trainer_utils import get_last_checkpoint

TRAIN_BATCH_SIZE = 24    # input batch size for training (default: 64), 20 for fp32
VALID_BATCH_SIZE = 64    # input batch size for testing (default: 1000)
TRAIN_EPOCHS = 10        # number of epochs to train (default: 10)
LEARNING_RATE = 1e-4     # learning rate (default: 0.001)
WEIGHT_DECAY = 0.01
SEED = 42                # random seed (default: 42)
MAX_LEN = 512
VOCAB_SIZE = 32000

config = {
    "architectures": [
        "RobertaForMaskedLM"
    ],
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-05,
    "max_position_embeddings": 514,
    "model_type": "roberta",
    "num_attention_heads": 12,
    "num_hidden_layers": 6,
    "type_vocab_size": 1,
    "vocab_size": VOCAB_SIZE,
    "boost_mode": 0,
    "boost_factor": 0.0,
    "boost_prem_grp": 1.0,
    "boost_elayer": 0
}

def DataProcess():
    config["boost_factor"] = args.boost_factor
    print("boost_factor in config : %.2f\n" % config["boost_factor"])

    config_r = RobertaConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=config['max_position_embeddings'],
        num_attention_heads=config['num_attention_heads'],
        num_hidden_layers=config['num_hidden_layers'],
        type_vocab_size=config['type_vocab_size'],
        boost_factor=config["boost_factor"],
        boost_prem_grp=config["boost_prem_grp"],
        boost_elayer=config["boost_elayer"],
    )
    model = RobertaForMaskedLM(config=config_r)
    print('Num parameters: ', model.num_parameters())

    tokenizer = BertTokenizer.from_pretrained(tokenizer_folder)

    # paths = [str(x) for x in Path(dataset_folder).glob("*.txt")]
    # path = ['k_wiki.txt', 'm_news_2020.txt', 'm_news_2021.txt', 'm_news.txt', 'm_written.txt']
    path = ['m_news.txt', 'm_written.txt']

    paths, paths_eval = [], []
    for v in path:
        paths.append(dataset_folder + '/' + v)
        paths_eval.append(dataset_folder + '/' + v.replace('.txt', '_eval.txt'))

    train_dataset = load_dataset('text', data_files = paths, split='train')
    eval_dataset = load_dataset('text', data_files = paths_eval, split='train')
    train_dataset = train_dataset.shuffle(seed=2022)
    eval_dataset = eval_dataset.shuffle(seed=2022)
    if args.clear_cache != "no":
        print("clearing cache...")
        train_dataset.cleanup_cache_files()
        eval_dataset.cleanup_cache_files()
    
    train_dataset = train_dataset.map(lambda ex: tokenizer(ex["text"], max_length = MAX_LEN, truncation=True, padding=True), num_proc=32) #, batched=True, batch_size=TRAIN_BATCH_SIZE)
    eval_dataset = eval_dataset.map(lambda ex: tokenizer(ex["text"], max_length = MAX_LEN, truncation=True, padding=True), num_proc=8) #, batched=True, batch_size=VALID_BATCH_SIZE)

    # to use --ignore_data_skip. if not, it takes too long time to get the exact position of the previous epoch.
    # Do not feed the argument of seed. The data should be different at every epoch.
    train_dataset = train_dataset.shuffle()
    eval_dataset = eval_dataset.shuffle()

    columns_to_return = ['input_ids', 'attention_mask', 'boost_mask']
    train_dataset.set_format(type='torch', columns=columns_to_return)
    eval_dataset.set_format(type='torch', columns=columns_to_return)
    print("\ntrainset_len: %d, evalset_len: %d" %(len(train_dataset), len(eval_dataset)))
    print(train_dataset[0])
    print(eval_dataset[0])

    return model, tokenizer, train_dataset, eval_dataset

def main():
    # Define the Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=ckpt_folder,
        overwrite_output_dir=True,
        save_strategy = 'steps',
        evaluation_strategy = 'steps',
        num_train_epochs=TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=TRAIN_BATCH_SIZE, 
        per_device_eval_batch_size=VALID_BATCH_SIZE, 
        save_steps=200,
        logging_strategy = 'steps',
        logging_steps=10,    # 10                   ####
        eval_steps=1200,     # 800
        gradient_checkpointing=False,
        fp16=True,                                  ####
        gradient_accumulation_steps=86,             #### 64, 113315494 * 10 epoch / (batch 24 * 86 * 4 gpu)
        eval_accumulation_steps=64,
        disable_tqdm=False,
        log_on_each_node=True,
        ignore_data_skip=True,
        save_total_limit=5,
        do_train=True,
        ddp_find_unused_parameters=False
    )

    # Create the trainer for our model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        #prediction_loss_only=True,
    )

    # Detecting last checkpoint.
    if args.resume_from_checkpoint == 'yes':
        training_args.overwrite_output_dir = False

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        print("last checkpoint : %s" % last_checkpoint)
    checkpoint = None
    if last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Train the model
    print("\nstart training...")
    trainer.train(resume_from_checkpoint=checkpoint)

    if not os.path.isdir(model_folder) :                             # make it if not exist 
        os.mkdir(model_folder)
    trainer.save_model(model_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear_cache', default="no", type=str, help='clear all the cached data')
    parser.add_argument('--resume_from_checkpoint', default="yes", type=str, help='resume_from_checkpoint')
    parser.add_argument("--boost_factor", default=0.0, type=float, help="boost factor")               #

    print = functools.partial(print, flush=True)

    args = parser.parse_args()
    print("\nroberta_s_train2.py : clear_cache=%s, resume_from_checkpoint=%s, boost_factor=%.2f" \
          % (args.clear_cache, args.resume_from_checkpoint, args.boost_factor))

    n_gpu = torch.cuda.device_count()
    print("number of gpus : %d" % n_gpu)

    #Set the path to the data folder, datafile and output folder and files
    root_folder = '/home/jongwonkim/test/paper/pretraining/'
    dataset_folder = os.path.abspath(os.path.join(root_folder, '../text_corpus'))
    model_folder = os.path.abspath(os.path.join(root_folder, 'roberta'))
    ckpt_folder = os.path.abspath(os.path.join(root_folder, 'ckpt/rs'))
    tokenizer_folder= os.path.abspath(os.path.join(root_folder, 'tokenizer'))

    ckpt_folder += "_bf%.1f" % args.boost_factor
    print("ckpt_folder : %s" % ckpt_folder)
    if not os.path.isdir(ckpt_folder) :                             # make it if not exist 
        os.mkdir(ckpt_folder)
        args.resume_from_checkpoint = 'no'

    start = time.time()                                             # 시작 시간 저장
    model, tokenizer, train_dataset, eval_dataset = DataProcess()
    main()
    print("Elapsed time : %.1f min\n" %((time.time() - start)/60))  # 현재시각 - 시작시간 = 실행 시간
