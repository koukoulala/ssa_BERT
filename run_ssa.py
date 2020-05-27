# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import time
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from torch.utils.data import DataLoader,TensorDataset,Dataset
from torch.utils.data.sampler import  RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import copy
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling import BertForNSPAug, BertConfig, WEIGHTS_NAME, CONFIG_NAME, BertForSequenceClassification, BertForNSP_co
from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear
from data_util import *
from torch.nn import CrossEntropyLoss, MSELoss


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--data_dir",
                    default='./glue_data/',
                    type=str,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")
parser.add_argument("--task_name",
                    default='RTE',
                    type=str,
                    help="The name of the task to train.")
parser.add_argument("--output_dir",
                    default='./results',
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--ckpt",
                    default='',
                    type=str,
                    help="The checkpoint of models.")
parser.add_argument("--vocab_file",
                    default='./vocab.txt',
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--only_bert",
                    default=0,
                    type=int,
                    help="Only running bert.")

## Other parameters
parser.add_argument("--cache_dir",
                    default="./results/bert_models",
                    type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--max_seq_length",
                    default=128,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--do_train",
                    default=True,
                    action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_eval",
                    default=True,
                    action='store_true',
                    help="Whether to run eval on the dev set.")
parser.add_argument("--do_test",
                    default=True,
                    action='store_true',
                    help="Whether to run test on the test set.")
parser.add_argument("--do_lower_case",
                    default=True,
                    action='store_true',
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--train_batch_size",
                    default=32,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size",
                    default=32,
                    type=int,
                    help="Total batch size for eval.")
parser.add_argument("--learning_rate",
                    default=2e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",
                    default=3.0,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument('--fp16',
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale',
                    type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                         "0 (default value): dynamic loss scaling.\n"
                         "Positive power of 2: static loss scaling value.\n")
parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
parser.add_argument("--ckpt_cache_dir",
                    default="./results/ckpt_bert_models",
                    type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--cls_weight",
                    default=0.5,
                    type=float,
                    help="The weight of [CLS] during linear combine operation in SSA.")
parser.add_argument("--attention_threshold",
                    default=0.3,
                    type=float,
                    help="Attention value above this threshold is used for computation")
parser.add_argument("--aug_loss_weight",
                    default=0.3,
                    type=float)
parser.add_argument("--num_show",
                    default=5,
                    type=int,
                    help="The number of showing examples.")
parser.add_argument("--aug_threshold",
                    default=0.1,
                    type=float,
                    help="The threshold of doing augmentation")
parser.add_argument("--rm_threshold",
                    default=0.7,
                    type=float,
                    help="The threshold of masking tokens")
parser.add_argument("--use_saved",
                    default=0,
                    type=int,
                    help="Whether to use saved bert_ckpt.")
parser.add_argument("--co_training",
                    default=False,
                    action='store_true',
                    help="Whether to do co_training.")
parser.add_argument("--share_weight",
                    default=1,
                    type=int,
                    help="Whether to share weight.")
parser.add_argument("--double_ori",
                    default=0,
                    type=int,
                    help="Whether to double original data.")
parser.add_argument("--aug_ratio_each",
                    default=0.2,
                    type=float,
                    help="The mask ration of each epoch")
parser.add_argument("--do_first_eval",
                    default=0,
                    type=int,
                    help="Whether to do first test.")


args = parser.parse_args()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}

    else:
        raise KeyError(task_name)

def accuracy(out, labels, type="seq", task_name="others"):
    if type == "seq":
        if task_name == "others":
            outputs = np.argmax(out, axis=1)
            res = np.sum(outputs == labels)
            return res
        else:
            return compute_metrics(task_name,out,labels)
    else:
        res = 0
        outputs = np.argmax(out, axis=2)
        num_tokens = 0
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i][j] != -1:
                    num_tokens += 1
                    if outputs[i][j] == labels[i][j]:
                        res += 1

        return res,num_tokens


def main(args):
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        logger.info("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    args.data_dir = os.path.join(args.data_dir, args.task_name)
    args.output_dir = os.path.join(args.output_dir, args.task_name)
    logger.info("args = %s", args)

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "wnli": WnliProcessor,
    }

    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification",
        "wnli": "classification",
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        logger.info("Output directory already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except:
            pass
            logger.info("catch a error")

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.vocab_file, do_lower_case=args.do_lower_case)

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(PYTORCH_PRETRAINED_BERT_CACHE,
                                                                   'distributed_{}'.format(args.local_rank))

    # use bert to aug train_examples
    ori_train_examples = processor.get_train_examples(args.data_dir)
    eval_examples = processor.get_dev_examples(args.data_dir)

    if args.double_ori == 0:
        num_train_optimization_steps = int(
            len(ori_train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    else:
        num_train_optimization_steps = int(
            len(ori_train_examples) * 2 / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()


    if args.use_saved == 1:
        bert_saved_dir = args.ckpt
        if args.co_training:
            model = BertForNSP_co.from_pretrained(bert_saved_dir,
                                                  cache_dir=args.ckpt_cache_dir,
                                                  num_labels=num_labels,
                                                  args=args)
        elif args.only_bert:
            model = BertForSequenceClassification.from_pretrained(bert_saved_dir,
                                                                  cache_dir=args.ckpt_cache_dir,
                                                                  num_labels=num_labels)
        else:
            model = BertForNSPAug.from_pretrained(bert_saved_dir,
                                              cache_dir=args.ckpt_cache_dir,
                                              num_labels=num_labels,
                                              args=args)
    else:
        if args.only_bert:
            model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                                  cache_dir=cache_dir,
                                                                  num_labels=num_labels)
        else:
            model = BertForNSPAug.from_pretrained(args.bert_model,
                                              cache_dir=cache_dir,
                                              num_labels=num_labels,
                                              args=args)
    model.cuda()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_first_eval:
        args.do_train = False
        res_file = os.path.join(args.output_dir, "first_test.tsv")

        eval_loss, eval_seq_loss, eval_aug_loss, eval_res, eval_aug_accuracy, res_parts = \
            do_evaluate(args, processor, label_list, tokenizer, model, 0, output_mode, num_labels, task_name,
                        eval_examples, type="dev")
        eval_res.update(res_parts)

        for key in sorted(eval_res.keys()):
            logger.info("first evaluation:  %s = %s", key, str(eval_res[key]))

        idx, preds = do_test(args, label_list, task_name, processor, tokenizer, output_mode, model)

        dataframe = pd.DataFrame({'index': range(idx), 'prediction': preds})
        dataframe.to_csv(res_file, index=False, sep='\t')
        logger.info("  Num test length = %d", idx)
        logger.info("  Done ")

        # write mm test results
        if task_name == "mnli":
            res_file = os.path.join(args.output_dir,
                                    "first_test_mm.tsv")

            idx, preds = do_test(args, label_list, task_name, processor, tokenizer, output_mode, model, do_mm=True)

            dataframe = pd.DataFrame({'index': range(idx), 'prediction': preds})
            dataframe.to_csv(res_file, index=False, sep='\t')
            logger.info("  Num test length = %d", idx)
            logger.info("  Done write mm")


    if args.do_train:
        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

        global_step = 0
        best_val_acc = 0.0
        first_time = time.time()

        logger.info("***** Running training *****")
        logger.info("  Num original examples = %d", len(ori_train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        model.train()
        aug_ratio = 0.0
        aug_seed = np.random.randint(0, 1000)
        for epoch in range(int(args.num_train_epochs)):
            if args.only_bert:
                train_features = convert_examples_to_features(ori_train_examples, label_list, args.max_seq_length,
                                tokenizer, num_show=args.num_show, output_mode=output_mode, args=args)
            else:
                logger.info("epoch=%d,  aug_ratio = %f,  aug_seed=%d", epoch, aug_ratio, aug_seed)
                train_examples = Aug_each_ckpt(ori_train_examples, label_list, model, tokenizer, args=args,
                                             num_show=args.num_show, output_mode=output_mode, seed=aug_seed,
                                             aug_ratio=aug_ratio, use_bert=False)
                if aug_ratio + args.aug_ratio_each < 1.0:
                    aug_ratio += args.aug_ratio_each
                aug_seed += 1

                train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer,
                                                              num_show=args.num_show, output_mode=output_mode, args=args)
            logger.info("Done convert features")
            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            if output_mode == "classification":
                all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
            elif output_mode == "regression":
                all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

            token_real_label = torch.tensor([f.token_real_label for f in train_features], dtype=torch.long)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, token_real_label)
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_data)
            else:
                train_sampler = DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

            logger.info("begin training")
            tr_loss, tr_seq_loss, tr_aug_loss, train_seq_accuracy, train_aug_accuracy = 0, 0, 0, 0, 0
            nb_tr_examples, nb_tr_steps, nb_tr_tokens = 0, 0, 0
            preds = []
            all_labels = []
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.cuda() for t in batch)
                input_ids, input_mask, segment_ids, label_ids, token_real_label = batch
                if args.only_bert:
                    seq_logits = model(input_ids, segment_ids, input_mask, labels=None)
                else:
                    seq_logits, aug_logits, aug_loss = model(input_ids, segment_ids, input_mask, labels=None, token_real_label=token_real_label)
                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    seq_loss = loss_fct(seq_logits.view(-1, num_labels), label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    seq_loss = loss_fct(seq_logits.view(-1), label_ids.view(-1))

                token_real_label = token_real_label.detach().cpu().numpy()

                w = args.aug_loss_weight
                if args.only_bert:
                    loss = seq_loss
                else:
                    loss = (1 - w) * seq_loss + w * aug_loss

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10000.0)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                batch_loss = seq_loss.mean().item()
                tr_seq_loss += seq_loss.mean().item()
                seq_logits = seq_logits.detach().cpu().numpy()
                label_ids = label_ids.detach().cpu().numpy()
                if len(preds) == 0:
                    preds.append(seq_logits)
                    all_labels.append(label_ids)
                else:
                    preds[0] = np.append(preds[0], seq_logits, axis=0)
                    all_labels[0] = np.append(all_labels[0], label_ids,axis=0)

                if args.only_bert == 0:
                    aug_logits = aug_logits.detach().cpu().numpy()
                    tmp_train_aug_accuracy, tmp_tokens = accuracy(aug_logits, token_real_label, type="aug")
                    train_aug_accuracy += tmp_train_aug_accuracy
                    nb_tr_tokens += tmp_tokens
                    tr_aug_loss += aug_loss.mean().item()

                if global_step % 20 == 0:
                    loss = tr_loss / nb_tr_steps
                    seq_loss = tr_seq_loss / nb_tr_steps
                    aug_loss = tr_aug_loss / nb_tr_steps
                    tmp_pred = preds[0]
                    tmp_labels = all_labels[0]
                    if output_mode == "classification":
                        tmp_pred = np.argmax(tmp_pred, axis=1)
                    elif output_mode == "regression":
                        tmp_pred = np.squeeze(tmp_pred)
                    res = accuracy(tmp_pred, tmp_labels, task_name=task_name)

                    if nb_tr_tokens != 0:
                        aug_avg = train_aug_accuracy / nb_tr_tokens
                    else:
                        aug_avg = 0.0
                    log_string = ""
                    log_string += "epoch={:<5d}".format(epoch)
                    log_string += " step={:<9d}".format(global_step)
                    log_string += " total_loss={:<9.7f}".format(loss)
                    log_string += " seq_loss={:<9.7f}".format(seq_loss)
                    log_string += " aug_loss={:<9.7f}".format(aug_loss)
                    log_string += " batch_loss={:<9.7f}".format(batch_loss)
                    log_string += " lr={:<9.7f}".format(optimizer.get_lr()[0])
                    log_string += " |g|={:<9.7f}".format(total_norm)
                    #log_string += " tr_seq_acc={:<9.7f}".format(seq_avg)
                    log_string += " tr_aug_acc={:<9.7f}".format(aug_avg)
                    log_string += " mins={:<9.2f}".format(float(time.time() - first_time) / 60)
                    for key in sorted(res.keys()):
                        log_string += "  " + key + "= " + str(res[key])
                    logger.info(log_string)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            train_loss = tr_loss / nb_tr_steps

            if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and epoch % 1 == 0:
                tot_time = float(time.time() - first_time) / 60
                eval_loss, eval_seq_loss, eval_aug_loss, eval_res, eval_aug_accuracy, res_parts=\
                    do_evaluate(args, processor, label_list, tokenizer, model, epoch, output_mode, num_labels, task_name, eval_examples, type="dev")

                eval_res["tot_time"] = tot_time
                if "acc" in eval_res:
                    tmp_acc = eval_res["acc"]
                elif "mcc" in eval_res:
                    tmp_acc = eval_res["mcc"]
                else:
                    tmp_acc = eval_res["corr"]

                if tmp_acc >= best_val_acc:
                    best_val_acc = tmp_acc
                    dev_test = "dev"

                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_dir = os.path.join(args.output_dir, "dev_" + str(tmp_acc))
                    if not os.path.exists(output_model_dir):
                        os.makedirs(output_model_dir)
                    output_model_file = os.path.join(output_model_dir, WEIGHTS_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    output_config_file = os.path.join(output_model_dir, CONFIG_NAME)
                    with open(output_config_file, 'w') as f:
                        f.write(model_to_save.config.to_json_string())

                    result = {'eval_total_loss': eval_loss,
                              'eval_seq_loss': eval_seq_loss,
                              'eval_aug_loss': eval_aug_loss,
                              'eval_aug_accuracy': eval_aug_accuracy,
                              'global_step': global_step,
                              'train_loss': train_loss,
                              'best_epoch': epoch,
                              'train_batch_size': args.train_batch_size,
                              'args': args}

                    result.update(eval_res)
                    result.update(res_parts)

                    output_eval_file = os.path.join(args.output_dir,
                                                    dev_test + "_results_" + str(tmp_acc) + ".txt")
                    with open(output_eval_file, "w") as writer:
                        logger.info("***** Test results *****")
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key])))

                    # write test results
                    if args.do_test:
                        res_file = os.path.join(args.output_dir,
                                                    "test_" + str(tmp_acc)+".tsv")

                        idx, preds = do_test(args, label_list, task_name, processor, tokenizer, output_mode, model)

                        dataframe = pd.DataFrame({'index': range(idx), 'prediction': preds})
                        dataframe.to_csv(res_file, index=False, sep='\t')
                        logger.info("  Num test length = %d", idx)
                        logger.info("  Done ")

                        # write mm test results
                        if task_name == "mnli":
                            res_file = os.path.join(args.output_dir,
                                                    "test_mm_" + str(tmp_acc) + ".tsv")

                            idx, preds = do_test(args, label_list, task_name, processor, tokenizer, output_mode, model, do_mm=True)

                            dataframe = pd.DataFrame({'index': range(idx), 'prediction': preds})
                            dataframe.to_csv(res_file, index=False, sep='\t')
                            logger.info("  Num test length = %d", idx)
                            logger.info("  Done write mm")

                else:
                    logger.info("  tmp_val_acc = %f", tmp_acc)


def do_evaluate(args, processor, label_list, tokenizer, model, epoch, output_mode, num_labels, task_name, eval_examples, type="dev"):

    nb_eval_steps, nb_eval_examples, nb_eval_tokens = 0, 0, 0
    eval_loss, eval_seq_loss, eval_aug_loss, eval_seq_accuracy, eval_aug_accuracy = 0, 0, 0, 0, 0
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, type=type, num_show=args.num_show,
        output_mode=output_mode, args=args)
    logger.info("  Num examples = %d", len(eval_examples))

    max_n = 1
    res_parts = {}
    if type == "dev":
        max_n = 3
        random.shuffle(eval_features)
        eval_all_fe = eval_features
        inter = int(len(eval_all_fe) / 3)
        eval_loss_all, eval_seq_loss_all, eval_aug_loss_all, eval_aug_accuracy_all = 0.0, 0.0, 0.0, 0.0
    for kk in range(max_n):
        # logger.info("inter",inter,kk,kk*inter,(kk+1)*inter)
        if type == "dev":
            eval_features = eval_all_fe[kk * inter:(kk + 1) * inter]
        logger.info("***** Running %s *****", type)
        logger.info("  Num dev examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

        token_real_label = torch.tensor([f.token_real_label for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, token_real_label)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        preds = []
        all_labels = []
        for batch in eval_dataloader:
            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, label_ids, token_real_label = batch

            with torch.no_grad():
                if args.only_bert:
                    seq_logits = model(input_ids, segment_ids, input_mask, labels=None)
                else:
                    seq_logits, aug_logits, aug_loss = model(input_ids, segment_ids, input_mask, labels=None,
                                                             token_real_label=token_real_label)
                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    seq_loss = loss_fct(seq_logits.view(-1, num_labels), label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    seq_loss = loss_fct(seq_logits.view(-1), label_ids.view(-1))
                w = args.aug_loss_weight
                if args.only_bert:
                    loss = seq_loss
                else:
                    loss = (1 - w) * seq_loss + w * aug_loss

            seq_logits = seq_logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()
            if len(preds) == 0:
                preds.append(seq_logits)
                all_labels.append(label_ids)
            else:
                preds[0] = np.append(preds[0], seq_logits, axis=0)
                all_labels[0] = np.append(all_labels[0], label_ids, axis=0)

            eval_seq_loss += seq_loss.mean().item()

            if args.only_bert == 0:
                aug_logits = aug_logits.detach().cpu().numpy()
                token_real_label = token_real_label.detach().cpu().numpy()
                tmp_eval_aug_accuracy, tmp_tokens = accuracy(aug_logits, token_real_label, type="aug")
                eval_aug_accuracy += tmp_eval_aug_accuracy
                nb_eval_tokens += tmp_tokens
                eval_aug_loss += aug_loss.mean().item()

            eval_loss += loss.mean().item()
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

            if nb_eval_steps % 10 == 0:
                loss = eval_loss / nb_eval_steps
                seq_loss = eval_seq_loss / nb_eval_steps
                aug_loss = eval_aug_loss / nb_eval_steps
                tmp_pred = preds[0]
                tmp_labels = all_labels[0]
                if output_mode == "classification":
                    tmp_pred = np.argmax(tmp_pred, axis=1)
                elif output_mode == "regression":
                    tmp_pred = np.squeeze(tmp_pred)
                res = accuracy(tmp_pred, tmp_labels, task_name=task_name)

                if nb_eval_tokens != 0:
                    aug_avg = eval_aug_accuracy / nb_eval_tokens
                else:
                    aug_avg = 0.0
                log_string = ""
                log_string += "epoch={:<5d}".format(epoch)
                log_string += " total_loss={:<9.7f}".format(loss)
                log_string += " seq_loss={:<9.7f}".format(seq_loss)
                log_string += " aug_loss={:<9.7f}".format(aug_loss)
                #log_string += " valid_seq_acc={:<9.7f}".format(seq_avg)
                log_string += " valid_aug_acc={:<9.7f}".format(aug_avg)
                for key in sorted(res.keys()):
                    log_string += "  "+key+"= "+str(res[key])
                logger.info(log_string)


        eval_loss = eval_loss / nb_eval_steps
        eval_seq_loss = eval_seq_loss / nb_eval_steps
        eval_aug_loss = eval_aug_loss / nb_eval_steps
        tmp_pred = preds[0]
        tmp_labels = all_labels[0]
        if output_mode == "classification":
            tmp_pred = np.argmax(tmp_pred, axis=1)
        elif output_mode == "regression":
            tmp_pred = np.squeeze(tmp_pred)
        res = accuracy(tmp_pred, tmp_labels, task_name=task_name)

        if nb_eval_tokens != 0:
            eval_aug_accuracy = eval_aug_accuracy / nb_eval_tokens
        else:
            eval_aug_accuracy = 0.0

        if type == "dev":
            eval_loss_all += eval_loss
            eval_seq_loss_all += eval_seq_loss
            eval_aug_loss_all += eval_aug_loss
            eval_aug_accuracy_all += eval_aug_accuracy
            for key in res:
                res_parts[key + "_" + str(kk)] = res[key]
            if kk == 0:
                res_all=res
            else:
                for key in res:
                    res_all[key]+=res[key]

    if type == "dev":
        eval_loss = eval_loss_all/3.0
        eval_seq_loss = eval_seq_loss_all/3.0
        eval_aug_loss = eval_aug_loss_all/3.0
        eval_aug_accuracy = eval_aug_accuracy_all/3.0
        for key in res_all:
            res[key] = res_all[key]/3.0

    return eval_loss, eval_seq_loss, eval_aug_loss, res, eval_aug_accuracy, res_parts

def do_test(args, label_list, task_name, processor, tokenizer, output_mode, model, do_mm=False):
    label_map = {i: label for i, label in enumerate(label_list)}
    if do_mm:
        test_w_examples = processor.get_mm_test_examples(args.data_dir)
    else:
        test_w_examples = processor.get_test_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        test_w_examples, label_list, args.max_seq_length, tokenizer, type="test",
        num_show=args.num_show, output_mode=output_mode, args=args)
    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", len(test_w_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    token_real_label = torch.tensor([f.token_real_label for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, token_real_label)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    preds = []
    idx = 0
    for batch in eval_dataloader:
        batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, segment_ids, token_real_label = batch
        with torch.no_grad():
            if args.only_bert:
                seq_logits = model(input_ids, segment_ids, input_mask, labels=None)
            else:
                seq_logits, aug_logits, aug_loss = model(input_ids, segment_ids, input_mask, labels=None,
                                                     token_real_label=token_real_label)

            seq_logits = seq_logits.detach().cpu().numpy()

            if output_mode == "classification":
                outputs = np.argmax(seq_logits, axis=1)
                for i in range(outputs.shape[0]):
                    pred_label = label_map[outputs[i]]
                    preds.append(pred_label)
                    idx += 1

            elif output_mode == "regression":
                outputs = np.squeeze(seq_logits)
                for i in range(outputs.shape[0]):
                    preds.append(outputs[i])
                    idx += 1

    return idx, preds

if __name__ == "__main__":
    main(args)
