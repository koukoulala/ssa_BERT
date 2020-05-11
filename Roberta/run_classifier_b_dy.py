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

from configuration_roberta import RobertaConfig
from modeling_roberta import RobertaForSequenceClassification, RobertaForNSP_co, RobertaForNSPAug
from tokenization_roberta import RobertaTokenizer
from optimization import AdamW, get_linear_schedule_with_warmup

from data_util import *
from torch.nn import CrossEntropyLoss, MSELoss
from evaluation import get_score

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--data_dir",
                    default='/data/kou/glue_data/',
                    type=str,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--task_name",
                    default='MRPC',
                    type=str,
                    help="The name of the task to train.")
parser.add_argument("--output_dir",
                    default='./results_Roberta',
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--Roberta_saved_dir",
                    default='./results_Roberta',
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--vocab_file",
                    default='./vocab.txt',
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--model_name_or_path", default="roberta-base", type=str,
                        help="Path to pre-trained model or shortcut name selected ")


## Other parameters
parser.add_argument("--cache_dir",
                    default="./results_Roberta/Roberta_models",
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
parser.add_argument("--max_aug_n",
                    default=5,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--ckpt_cache_dir",
                    default="./results/ckpt_bert_models",
                    type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--cls_weight",
                    default=0.5,
                    type=float,
                    help="w")
parser.add_argument("--attention_threshold",
                    default=0.5,
                    type=float,
                    help="k")
parser.add_argument("--aug_weight",
                    default=1.0,
                    type=float,
                    help="weight")
parser.add_argument("--aug_loss_weight",
                    default=0.3,
                    type=float)
parser.add_argument("--search_hparam",
                    default=False,
                    action='store_true',
                    help="Whether to search_hparam.")
parser.add_argument("--conf_file",
                    default='./conf.json',
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--log_dir",
                    default='./log/search_hparam',
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--do_softmax",
                    default=1,
                    type=int,
                    help="Whether to do softmax.")
parser.add_argument("--available_gpus",
                    default='1',
                    type=str,
                    help="available_gpus")
parser.add_argument("--case_study",
                    default=False,
                    action='store_true',
                    help="Whether to do case_study.")
parser.add_argument("--share_weight",
                    default=1,
                    type=int,
                    help="Whether to do softmax.")
parser.add_argument("--num_show",
                    default=5,
                    type=int,
                    help="Whether to show examples.")
parser.add_argument("--aug_threshold",
                    default=0.1,
                    type=float,
                    help="k")
parser.add_argument("--rm_threshold",
                    default=0.9,
                    type=float,
                    help="k")
parser.add_argument("--do_aug_epoch",
                    default=1,
                    type=int,
                    help="Whether to do aug in every epoch.")
parser.add_argument("--use_saved",
                    default=1,
                    type=int,
                    help="Whether to do aug in every epoch.")
parser.add_argument("--use_stilts",
                    default=False,
                    action='store_true',
                    help="Whether to use_stilts checkpoint.")
parser.add_argument("--stilts_checkpoint",
                    default='./stilts_checkpoints',
                    type=str,
                    help="The directory of stilts checkpoint.")
parser.add_argument("--BERT_ALL_DIR", default='./cache/bert_metadata', type=str)
parser.add_argument("--do_first_eval",
                    default=False,
                    action='store_true',
                    help="Whether to do aug in every epoch.")
parser.add_argument("--do_add",
                    default=0,
                    type=int,
                    help="Whether to do add.")
parser.add_argument("--co_training",
                    default=False,
                    action='store_true',
                    help="Whether to do co_training.")
parser.add_argument("--change_generate",
                    default=False,
                    action='store_true',
                    help="Whether to do change_generate.")
parser.add_argument("--change_label",
                    default=0,
                    type=int,
                    help="Whether to change_label.")
parser.add_argument("--do_whole_word_mask",
                    default=0,
                    type=int,
                    help="Whether to do_whole_word_mask.")
parser.add_argument("--mask_token",
                    default=1,
                    type=int,
                    help="Whether to mask.")

parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--warmup_rate", default=0.06, type=float, help="Linear warmup over warmup_steps.")


args = parser.parse_args()


TF_PYTORCH_BERT_NAME_MAP = {
    "bert-base-uncased": "uncased_L-12_H-768_A-12",
    "bert-large-uncased": "uncased_L-24_H-1024_A-16",
}


def get_bert_config_path(bert_model_name,BERT_ALL_DIR):
    return os.path.join(BERT_ALL_DIR, TF_PYTORCH_BERT_NAME_MAP[bert_model_name],'bert_config.json')


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
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}

    elif task_name == "snli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "sst":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "cornell":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "figure":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "uci":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "twitter":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "text":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "imdb":
        return {"acc": simple_accuracy(preds, labels)}

    elif task_name == "sentihood_nli_b":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "sentihood_qa_b":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "semeval_nli_b":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "semeval_qa_b":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "test_sentihood":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

def accuracy(out, labels, type="seq", task_name="others"):
    if type=="seq":
        if task_name=="others":
            outputs = np.argmax(out, axis=1)
            res=np.sum(outputs == labels)
            return res
        else:
            return compute_metrics(task_name,out,labels)
    else:
        res=0
        outputs = np.argmax(out, axis=2)
        num_tokens=0
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i][j]!=-1:
                    num_tokens+=1
                    if outputs[i][j] == labels[i][j]:
                        res+=1

        return res,num_tokens

def main(args):
    report_metric = [0.0, 0.0, 0.0, 0.0]
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    args.data_dir=os.path.join(args.data_dir,args.task_name)
    args.output_dir=os.path.join(args.output_dir,args.task_name)
    args.Roberta_saved_dir = os.path.join(args.Roberta_saved_dir, args.task_name)
    logger.info("args = %s", args)

    processors = {

        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "wnli": WnliProcessor,

        "sentihood_nli_b": Sentihood_NLI_B_Processor,
        "sentihood_qa_b": Sentihood_QA_B_Processor,
        "semeval_nli_b": Semeval_NLI_B_Processor,
        "semeval_qa_b": Semeval_QA_B_Processor,
        "test_sentihood": test_Sentihood_Processor,
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

        "sentihood_nli_b": "classification",
        "sentihood_qa_b": "classification",
        "semeval_nli_b": "classification",
        "semeval_qa_b": "classification",
        "test_sentihood": "classification"
    }

    have_test = ["mrpc","sst-2"]
    absa_file = ["sentihood_nli_b", "sentihood_qa_b", "semeval_nli_b", "semeval_qa_b","test_sentihood"]

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
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except:
            print("catch a error")

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # use bert to aug train_examples
    ori_train_examples = processor.get_train_examples(args.data_dir)


    config_class, tokenizer_class = (RobertaConfig, RobertaTokenizer)

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None)

    if args.do_aug_epoch == 0:
        ckpt_file = os.path.join(args.Roberta_saved_dir,
                                 "dev_val_only_aug_False_only_seq_True_only_bert_True_with_attention_False")

        model_class = RobertaForSequenceClassification
        roberta_model = model_class.from_pretrained(ckpt_file)

        roberta_model.cuda()
        aug_seed = np.random.randint(0,1000)
        #right=max(2,min(3,args.max_aug_n))
        #num_no_aug = np.random.randint(1, right)
        num_no_aug = 1
        logger.info("  max_aug_n = %d", args.max_aug_n)
        logger.info("  num_no_aug = %d", num_no_aug)
        train_examples = Aug_by_epoch(ori_train_examples, label_list, roberta_model, tokenizer, args=args,
                                     num_show=args.num_show, output_mode=output_mode, seed=aug_seed,
                                     aug_n=args.max_aug_n, num_no_aug=num_no_aug, use_bert=False, do_roberta=1,
                                      ssa_roberta=0, pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0])

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    else:
        aug_n_list, num_no_aug_list = [], []
        num_repeat = 0
        for epoch in range(int(args.num_train_epochs)):
            aug_n = np.random.randint(2, args.max_aug_n)
            #right = max(2, min(3, aug_n))
            #num_no_aug = np.random.randint(1, right)
            num_no_aug = 1
            aug_n_list.append(aug_n)
            num_no_aug_list.append(num_no_aug)
            num_repeat += aug_n

        num_train_optimization_steps = int(
            len(
                ori_train_examples) * num_repeat / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.use_saved == 1:
        if args.co_training:
            ckpt_file = os.path.join(args.Roberta_saved_dir,
                                          "dev_val_only_aug_False_only_seq_True_only_bert_True_with_attention_False")

            model_class = RobertaForNSP_co
            model = model_class.from_pretrained(ckpt_file, args=args)

        else:
            ckpt_file = os.path.join(args.Roberta_saved_dir,
                                          "dev_val_only_aug_False_only_seq_True_only_bert_True_with_attention_False")

            model_class = RobertaForNSPAug
            model = model_class.from_pretrained(ckpt_file, args=args)
    else:
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                              num_labels=num_labels,
                                              finetuning_task=task_name,
                                              cache_dir=args.cache_dir if args.cache_dir else None)

        model_class = RobertaForNSPAug
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None,
                                            args=args
                                            )

    model.cuda()
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
        args.warmup_rate * num_train_optimization_steps),
                                                num_training_steps=num_train_optimization_steps)

    if args.do_first_eval:
        args.do_train = False
        eval_loss, eval_seq_loss, eval_aug_loss, eval_res, eval_aug_accuracy, res_parts = \
            do_evaluate(args, processor, label_list, tokenizer, model, 0, output_mode, num_labels, task_name,
                        type="dev")
        if task_name in have_test and args.do_test:
            logger.info("***** Running testing *****")
            eval_loss, eval_seq_loss, eval_aug_loss, eval_res, eval_aug_accuracy, _ = \
                do_evaluate(args, processor, label_list, tokenizer, model, 0, output_mode, num_labels,
                            task_name, type="test")

        if "acc" in eval_res:
            re_me = eval_res["acc"]
        elif "mcc" in eval_res:
            re_me = eval_res["mcc"]
        else:
            re_me = eval_res["corr"]
        output_eval_file = os.path.join(args.output_dir, "start_roberta_results_" + str(re_me) + "_dy.txt")

        res_file = os.path.join(args.output_dir,
                                "start_roberta_B_test_results_" + str(re_me) + "_dy.tsv")

        f_absa = None

        f_absa, idx, preds = do_test(args, label_list, task_name, processor, tokenizer, output_mode, model,
                                     absa_file,
                                     f_absa=f_absa)

        dataframe = pd.DataFrame({'index': range(idx), 'prediction': preds})
        dataframe.to_csv(res_file, index=False, sep='\t')
        logger.info("  Num test length = %d", idx)
        logger.info("  Done ")

        # write mm test results
        if task_name == "mnli":
            res_file = os.path.join(args.output_dir,
                                    "start_roberta_mm_results_b_" + str(re_me) + "_dy.tsv")

            _, idx, preds = do_test(args, label_list, task_name, processor, tokenizer, output_mode,
                                    model, absa_file, do_mm=True)

            dataframe = pd.DataFrame({'index': range(idx), 'prediction': preds})
            dataframe.to_csv(res_file, index=False, sep='\t')
            logger.info("  Num test length = %d", idx)
            logger.info("  Done write mm")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval first results *****")
            for key in sorted(eval_res.keys()):
                logger.info("  %s = %s", key, str(eval_res[key]))
                writer.write("%s = %s\n" % (key, str(eval_res[key])))


    global_step = 0
    best_val_acc=0.0
    first_time = time.time()
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num original examples = %d", len(ori_train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        if args.do_aug_epoch == 0:
            train_features = convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, tokenizer, aug_n=1, num_show=args.num_show,
                output_mode=output_mode,args=args, change_generate=args.change_generate,
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0], do_roberta=1)
        model.train()
        for epoch in range(int(args.num_train_epochs)):
            if args.do_aug_epoch == 1:
                logger.info("  max_aug_n = %d", aug_n_list[epoch])
                logger.info("  num_no_aug = %d", num_no_aug_list[epoch])

                aug_seed = np.random.randint(0, 1000)
                train_examples = Aug_by_epoch(ori_train_examples, label_list, model, tokenizer, args=args,
                                             num_show=args.num_show, output_mode=output_mode, seed=aug_seed,
                                             aug_n=aug_n_list[epoch], num_no_aug=num_no_aug_list[epoch], use_bert=False,
                                              change_generate=args.change_generate, do_roberta=1, ssa_roberta=1,
                                      pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0])
                train_features = convert_examples_to_features(
                    train_examples, label_list, args.max_seq_length, tokenizer, aug_n=1, num_show=args.num_show,
                    output_mode=output_mode, args=args, change_generate=args.change_generate,
                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0], do_roberta=1)

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

            tr_loss,tr_seq_loss,tr_aug_loss, train_seq_accuracy, train_aug_accuracy = 0, 0, 0, 0, 0
            nb_tr_examples, nb_tr_steps, nb_tr_tokens = 0, 0, 0
            preds = []
            all_labels=[]
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.cuda() for t in batch)
                input_ids, input_mask, segment_ids, label_ids, token_real_label = batch
                seq_logits, aug_logits, aug_loss = model(input_ids, input_mask, labels=None, token_real_label=token_real_label)
                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    seq_loss = loss_fct(seq_logits.view(-1, num_labels), label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    seq_loss = loss_fct(seq_logits.view(-1), label_ids.view(-1))
                w = args.aug_loss_weight
                total_loss = (1 - w) * seq_loss + w * aug_loss
                loss = total_loss
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10000.0)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                tr_seq_loss += seq_loss.mean().item()
                seq_logits = seq_logits.detach().cpu().numpy()
                label_ids = label_ids.detach().cpu().numpy()
                if len(preds) == 0:
                    preds.append(seq_logits)
                    all_labels.append(label_ids)
                else:
                    preds[0] = np.append(preds[0], seq_logits, axis=0)
                    all_labels[0]=np.append(all_labels[0],label_ids,axis=0)

                aug_logits = aug_logits.detach().cpu().numpy()
                token_real_label = token_real_label.detach().cpu().numpy()
                tmp_train_aug_accuracy,tmp_tokens = accuracy(aug_logits, token_real_label, type="aug")
                train_aug_accuracy += tmp_train_aug_accuracy
                nb_tr_tokens +=tmp_tokens
                tr_aug_loss += aug_loss.mean().item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    global_step += 1

                if global_step % 20 == 0:
                    loss = tr_loss / nb_tr_steps
                    seq_loss = tr_seq_loss / nb_tr_steps
                    aug_loss = tr_aug_loss / nb_tr_steps
                    tmp_pred = preds[0]
                    tmp_labels=all_labels[0]
                    if output_mode == "classification":
                        tmp_pred = np.argmax(tmp_pred, axis=1)
                    elif output_mode == "regression":
                        tmp_pred = np.squeeze(tmp_pred)
                    res = accuracy(tmp_pred, tmp_labels, task_name=task_name)

                    aug_avg = train_aug_accuracy / nb_tr_tokens
                    log_string = ""
                    log_string += "epoch={:<5d}".format(epoch)
                    log_string += " step={:<9d}".format(global_step)
                    log_string += " total_loss={:<9.7f}".format(loss)
                    log_string += " seq_loss={:<9.7f}".format(seq_loss)
                    log_string += " aug_loss={:<9.7f}".format(aug_loss)
                    log_string += " lr={:<9.7f}".format(scheduler.get_lr()[0])
                    log_string += " |g|={:<9.7f}".format(total_norm)
                    #log_string += " tr_seq_acc={:<9.7f}".format(seq_avg)
                    log_string += " tr_aug_acc={:<9.7f}".format(aug_avg)
                    log_string += " mins={:<9.2f}".format(float(time.time() - first_time) / 60)
                    for key in sorted(res.keys()):
                        log_string += "  "+key+"= "+str(res[key])
                    logger.info(log_string)

            train_loss = tr_loss / nb_tr_steps

            if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and epoch % 1 == 0:
                eval_loss,eval_seq_loss,eval_aug_loss,eval_res,eval_aug_accuracy,res_parts=\
                    do_evaluate(args,processor,label_list,tokenizer,model,epoch,output_mode,num_labels,task_name,type="dev")

                if "acc" in eval_res:
                    tmp_acc=eval_res["acc"]
                elif "mcc" in eval_res:
                    tmp_acc = eval_res["mcc"]
                else:
                    tmp_acc=eval_res["corr"]

                if tmp_acc>=best_val_acc:
                    best_val_acc=tmp_acc
                    dev_test = "dev"

                    if task_name in have_test and args.do_test:
                        # save validate results first
                        val_result = {'eval_total_loss': eval_loss,
                                      'eval_seq_loss' : eval_seq_loss,
                                      'eval_aug_loss' : eval_aug_loss,
                                      'eval_aug_accuracy' : eval_aug_accuracy,
                                      'global_step': global_step,
                                      'train_loss': train_loss,
                                      'best_epoch':epoch,
                                      'train_batch_size':args.train_batch_size,
                                      'max_aug_n':args.max_aug_n,
                                      'args':args}

                        val_result.update(eval_res)
                        val_result.update(res_parts)

                        logger.info("***** Running testing *****")
                        dev_test = "test"
                        eval_loss, eval_seq_loss, eval_aug_loss, eval_res, eval_aug_accuracy,_ = \
                            do_evaluate(args, processor, label_list, tokenizer, model, epoch, output_mode, num_labels,
                                        task_name, type="test")

                    if "acc" in eval_res:
                        report_metric[0] = eval_res["acc"]
                        report_metric[1] = round(eval_res["acc"], 3)
                        if task_name in have_test:
                            report_metric[2] = eval_res["acc"]
                        else:
                            tmp_list = [res_parts["acc_0"], res_parts["acc_1"], res_parts["acc_2"]]
                            report_metric[2] = -np.var(tmp_list)
                    elif "mcc" in eval_res:
                        report_metric[0] = eval_res["mcc"]
                        report_metric[1] = round(eval_res["mcc"], 3)
                        if task_name in have_test:
                            report_metric[2] = eval_res["mcc"]
                        else:
                            tmp_list = [res_parts["mcc_0"], res_parts["mcc_1"], res_parts["mcc_2"]]
                            report_metric[2] = -np.var(tmp_list)
                    else:
                        report_metric[0] = eval_res["corr"]
                        report_metric[1] = round(eval_res["corr"], 3)
                        if task_name in have_test:
                            report_metric[2] = eval_res["corr"]
                        else:
                            tmp_list = [res_parts["corr_0"], res_parts["corr_1"], res_parts["corr_2"]]
                            report_metric[2] = -np.var(tmp_list)

                    report_metric[3] = 0.0
                    print("report_metric", report_metric)

                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_dir = os.path.join(args.output_dir,"roberta_B_dy_" + str(report_metric[0]))
                    if not os.path.exists(output_model_dir):
                        try:
                            os.makedirs(output_model_dir)
                        except:
                            print("catch a error at output_model_dir")
                    model_to_save.save_pretrained(output_model_dir)
                    tokenizer.save_pretrained(output_model_dir)
                    output_model_file = os.path.join(output_model_dir, 'pytorch_model.bin')
                    torch.save(model_to_save.state_dict(), output_model_file)

                    result = {'eval_total_loss': eval_loss,
                              'eval_seq_loss' : eval_seq_loss,
                              'eval_aug_loss' : eval_aug_loss,
                              'eval_aug_accuracy' : eval_aug_accuracy,
                              'global_step': global_step,
                              'train_loss': train_loss,
                              'best_epoch':epoch,
                              'train_batch_size':args.train_batch_size,
                              'max_aug_n':args.max_aug_n,
                              'args':args}

                    result.update(eval_res)
                    if task_name not in have_test:
                        result.update(res_parts)

                    if task_name in have_test:
                        # save eval results
                        output_eval_file = os.path.join(args.output_dir,
                                                        "roberta_b_dy_results_" + str(report_metric[0]) + ".txt")
                        with open(output_eval_file, "w") as writer:
                            logger.info("***** Eval results *****")
                            for key in sorted(val_result.keys()):
                                logger.info("  %s = %s", key, str(val_result[key]))
                                writer.write("%s = %s\n" % (key, str(val_result[key])))

                    output_eval_file = os.path.join(args.output_dir,
                                                    dev_test + "_roberta_b_dy_results_" + str(report_metric[0]) + ".txt")
                    with open(output_eval_file, "w") as writer:
                        logger.info("***** Test results *****")
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key])))

                    # write test results
                    if args.do_test :
                        res_file=os.path.join(args.output_dir,
                                                    "roberta_B_dy_test_results_" +str(report_metric[0])+".tsv")

                        f_absa = None
                        if task_name in absa_file:
                            absa_path = os.path.join(args.output_dir, "absa_roberta_test_b_dy_" + str(report_metric[0]) + ".txt")
                            f_absa = open(absa_path, "w")

                        f_absa, idx, preds = do_test(args, label_list, task_name, processor, tokenizer, output_mode,
                                                     model, absa_file,
                                                     f_absa=f_absa)

                        if task_name in absa_file:
                            f_absa.close()
                            res_absa = get_score(task_name, absa_path, args.data_dir)
                            output_absa_file = os.path.join(args.output_dir,
                                                            "absa_roberta_results_b_dy_" + str(report_metric[0]) + ".txt")
                            with open(output_absa_file, "w") as writer:
                                logger.info("***** Eval results *****")
                                for key in sorted(res_absa.keys()):
                                    logger.info("  %s = %s", key, str(res_absa[key]))
                                    writer.write("%s = %s\n" % (key, str(res_absa[key])))
                        else:
                            dataframe = pd.DataFrame({'index': range(idx), 'prediction': preds})
                            dataframe.to_csv(res_file, index=False, sep='\t')
                        logger.info("  Num test length = %d", idx)
                        logger.info("  Done ")

                        # write mm test results
                        if task_name == "mnli":
                            res_file = os.path.join(args.output_dir,
                                                    "mm_roberta_results_b_dy_" + str(report_metric[0]) + ".tsv")

                            _, idx, preds = do_test(args, label_list, task_name, processor, tokenizer, output_mode,
                                                    model, absa_file, do_mm=True)

                            dataframe = pd.DataFrame({'index': range(idx), 'prediction': preds})
                            dataframe.to_csv(res_file, index=False, sep='\t')
                            logger.info("  Num test length = %d", idx)
                            logger.info("  Done write mm")

                else:
                    logger.info("  tmp_val_acc = %f", tmp_acc)

    if args.search_hparam:
        return report_metric


def do_evaluate(args,processor,label_list,tokenizer,model,epoch,output_mode,num_labels,task_name,type="dev"):

    nb_eval_steps, nb_eval_examples, nb_eval_tokens = 0, 0, 0
    eval_loss, eval_seq_loss, eval_aug_loss, eval_seq_accuracy, eval_aug_accuracy = 0, 0, 0, 0, 0
    if type=="dev":
        eval_examples = processor.get_dev_examples(args.data_dir)
    else:
        eval_examples = processor.get_test_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer,type=type,num_show=args.num_show,output_mode=output_mode,
        args=args, change_generate=args.change_generate, pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0], do_roberta=1)
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
        # print("inter",inter,kk,kk*inter,(kk+1)*inter)
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
        preds=[]
        all_labels=[]
        for batch in eval_dataloader:
            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, label_ids, token_real_label = batch

            with torch.no_grad():
                seq_logits, aug_logits, aug_loss = model(input_ids, input_mask, labels=None,
                                                             token_real_label=token_real_label)
                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    seq_loss = loss_fct(seq_logits.view(-1, num_labels), label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    seq_loss = loss_fct(seq_logits.view(-1), label_ids.view(-1))
                w = args.aug_loss_weight
                total_loss = (1 - w) * seq_loss + w * aug_loss
                loss = total_loss

            seq_logits = seq_logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()
            if len(preds) == 0:
                preds.append(seq_logits)
                all_labels.append(label_ids)
            else:
                preds[0] = np.append(preds[0], seq_logits, axis=0)
                all_labels[0] = np.append(all_labels[0], label_ids, axis=0)

            eval_seq_loss += seq_loss.mean().item()

            aug_logits = aug_logits.detach().cpu().numpy()
            token_real_label = token_real_label.detach().cpu().numpy()
            tmp_eval_aug_accuracy,tmp_tokens = accuracy(aug_logits, token_real_label, type="aug")
            eval_aug_accuracy += tmp_eval_aug_accuracy
            nb_eval_tokens +=tmp_tokens
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

                aug_avg = eval_aug_accuracy / nb_eval_tokens
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

        eval_aug_accuracy = eval_aug_accuracy / nb_eval_tokens

        if type == "dev":

            eval_loss_all+=eval_loss
            eval_seq_loss_all+=eval_seq_loss
            eval_aug_loss_all+=eval_aug_loss
            eval_aug_accuracy_all+=eval_aug_accuracy
            for key in res:
                res_parts[key + "_" + str(kk)] = res[key]
            if kk==0:
                res_all=res
            else:
                for key in res:
                    res_all[key]+=res[key]


    if type == "dev":
        eval_loss=eval_loss_all/3.0
        eval_seq_loss=eval_seq_loss_all/3.0
        eval_aug_loss=eval_aug_loss_all/3.0
        eval_aug_accuracy=eval_aug_accuracy_all/3.0
        for key in res_all:
            res[key] = res_all[key]/3.0

    return eval_loss,eval_seq_loss,eval_aug_loss,res,eval_aug_accuracy,res_parts

def do_test(args, label_list, task_name, processor, tokenizer, output_mode, model, absa_file,
             f_absa=None, do_mm=False, do_sig=False, only_seq=False):

    label_map = {i: label for i, label in enumerate(label_list)}
    if task_name == 'sst-2':
        eval_examples = processor.get_test_nolabel_examples(args.data_dir)
    elif do_mm:
        eval_examples = processor.get_mm_test_examples(args.data_dir)
    else:
        eval_examples = processor.get_test_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, type="test",
        num_show=args.num_show, output_mode=output_mode, args=args, change_generate=args.change_generate,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0], do_roberta=1)
    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", len(eval_examples))
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
            if only_seq:
                seq_logits = model(input_ids, segment_ids, input_mask, labels=None)
            else:
                seq_logits, aug_logits, aug_loss = model(input_ids, input_mask,
                                                     labels=None,
                                                     token_real_label=token_real_label)

            logits = F.softmax(seq_logits, dim=-1)
            logits = logits.detach().cpu().numpy()
            seq_logits = seq_logits.detach().cpu().numpy()
            if output_mode == "classification":
                outputs = np.argmax(seq_logits, axis=1)
                for i in range(outputs.shape[0]):
                    if task_name in absa_file:
                        f_absa.write(str(outputs[i]))
                        for ou in logits[i]:
                            f_absa.write(" " + str(ou))
                        f_absa.write("\n")
                    else:
                        pred_label = label_map[outputs[i]]
                        preds.append(pred_label)
                        idx += 1

            elif output_mode == "regression":
                outputs = np.squeeze(seq_logits)
                for i in range(outputs.shape[0]):
                    preds.append(outputs[i])
                    idx += 1

    return f_absa, idx, preds

if __name__ == "__main__":
    main(args)
