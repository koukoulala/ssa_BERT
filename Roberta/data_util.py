import pandas as pd
import csv
import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset,Dataset
from torch.utils.data.sampler import  RandomSampler, SequentialSampler
import random
import torch.nn.functional as F
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, rand_index_a=None, rand_index_b=None, is_effect=0):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.rand_index_a = rand_index_a
        self.rand_index_b = rand_index_b
        self.is_effect = is_effect


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, token_real_label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.token_real_label=token_real_label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, type = "train.tsv"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, type)), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, type = "train.tsv"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, type)), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_test_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")),
            "test_matched")

    def get_mm_test_examples(self, data_dir):
        """See base class."""
        return self._create_test_examples(
            self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")),
            "test_mismatched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_test_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[8]
            text_b = line[9]
            label = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir, type = "train.tsv"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, type)), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_test_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),"test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_test_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir, type = "train.tsv"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, type)), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),"test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_test_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=None))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir, type = "train.tsv"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, type)), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_test_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),"test")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_test_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=None))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir, type = "train.tsv"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, type)), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_test_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),"test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_test_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[1]
                text_b = line[2]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=None))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir, type = "train.tsv"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, type)), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_test_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),"test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_test_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=None))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir, type = "train.tsv"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, type)), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_test_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),"test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_test_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=None))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, type = "train.tsv"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, type)), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_test_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),"test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_test_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=None))
        return examples



def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, type="train", num_show=5,
                                 output_mode="classification", seed=42, args=None, pad_token=0, do_roberta=0):
    """Loads a data file into a list of `InputBatch`s."""

    np.random.seed(seed)
    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    ex_index=0
    for (_, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if example.rand_index_a != None:
            rand_index_a = example.rand_index_a
        else:
            rand_index_a = []
        if example.rand_index_b != None:
            rand_index_b = example.rand_index_b
        else:
            rand_index_b = []

        if example.text_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            if do_roberta:
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 4)
            else:
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3 )
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2 )]


        if do_roberta:
            tokens = ["<s>"] + tokens_a + ["</s>"]
        else:
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        token_real_label = [1] * len(tokens)
        token_real_label[0] = -1
        token_real_label[-1] = -1
        rand_index_a = [x + 1 for x in rand_index_a if x < len(tokens_a)]
        for idx in rand_index_a:
            token_real_label[idx] = 0

        if tokens_b:
            more_special = 0
            if do_roberta:
                tokens += ["</s>"] + tokens_b + ["</s>"]
                more_special = 1
            else:
                tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1 + more_special)

            token_real_label += [1] * (len(tokens_b) + 1 + more_special)
            token_real_label[-1] = -1
            rand_index_b = [x + len(tokens_a) + 2 + more_special for x in rand_index_b if x < len(tokens_b)]
            for idx in rand_index_b:
                token_real_label[idx] = 0

            #print("tokens_a,b", tokens,rand_index_a,rand_index_b,rand_token_a,rand_token_b,token_real_label)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        ignore_padding = [-1] * (max_seq_length - len(input_ids))
        input_ids += [pad_token] * (max_seq_length - len(input_ids))
        input_mask += padding
        segment_ids += padding
        token_real_label += ignore_padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(token_real_label) == max_seq_length

        if example.label==None:
            label_id = None
        else:
            if output_mode == "classification":
                label_id = label_map[example.label]
            elif output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(output_mode)

        if ex_index < num_show:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("token_real_label: %s" % " ".join([str(x) for x in token_real_label]))
            if type != "test":
                logger.info("sequence_label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              token_real_label=token_real_label))
        ex_index+=1

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def rm_sub(rand_idx, rm_index, tokens, num_rm, args):
    rm_list = []
    num_idx = 0
    tmp_idx = rand_idx
    while num_idx < num_rm:
        num_idx += 1
        rm_list.append(tmp_idx)
        tmp_idx = tmp_idx - 1

    rm_index.extend(rm_list)

    return rm_index

def rand_rm(rand_start,rand_end,rm_index,seed=42,args=None, tokens=None):
    #np.random.seed(seed)
    if rand_end-rand_start <= 1:
        return rm_index
    rand_idx = np.random.randint(rand_start, rand_end)
    p = np.random.rand()
    if p > args.rm_threshold and rand_idx >= rand_start + 2:
        rm_index = rm_sub(rand_idx, rm_index, tokens, 3, args)
    elif p > args.rm_threshold-(1.0-args.rm_threshold) and p < args.rm_threshold and rand_idx >= rand_start + 1:
        rm_index = rm_sub(rand_idx, rm_index, tokens, 2, args)
    elif rand_idx != None:
        rm_index = rm_sub(rand_idx, rm_index, tokens, 1, args)

    return rm_index

def aug_func(ori_examples, label_list, tokenizer, cnt, num_no_aug, args, seed, max_seq_length, output_mode, num_show,
             pad_token=0, do_roberta=0):

    label_map = {label: i for i, label in enumerate(label_list)}
    auged_examples = []
    features = []
    ex_index = 0
    for (guid, example) in enumerate(ori_examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = []
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
        label = example.label

        for i in range(cnt):
            rand_index_a, rand_index_b=[],[]
            rm_index_a, rm_index_b = [],[]
            if i >= num_no_aug:  # first several time is original
                per_array_a = np.array(range(len(tokens_a)))
                per_25_a = np.percentile(per_array_a, 25)
                per_50_a = np.percentile(per_array_a, 50)
                per_75_a = np.percentile(per_array_a, 75)
                for idx in range(len(tokens_a)):
                    p = np.random.rand()
                    if p > args.aug_threshold:
                        continue
                    else:
                        if idx < per_25_a:
                            rm_index_a = rand_rm(0, per_25_a, rm_index_a, seed=seed, args=args, tokens=tokens_a)
                        elif idx >= per_25_a and idx < per_50_a:
                            rm_index_a = rand_rm(per_25_a, per_50_a, rm_index_a, seed=seed, args=args, tokens=tokens_a)
                        elif idx >= per_50_a and idx < per_75_a:
                            rm_index_a = rand_rm(per_50_a, per_75_a, rm_index_a, seed=seed, args=args, tokens=tokens_a)
                        else:
                            rm_index_a = rand_rm(per_75_a, len(tokens_a), rm_index_a, seed=seed, args=args, tokens=tokens_a)

                if example.text_b:
                    per_array_b = np.array(range(len(tokens_b)))
                    per_25_b = np.percentile(per_array_b, 25)
                    per_50_b = np.percentile(per_array_b, 50)
                    per_75_b = np.percentile(per_array_b, 75)
                    for idx in range(len(tokens_b)):
                        p = np.random.rand()
                        if p > args.aug_threshold:
                            continue
                        else:
                            if idx < per_25_b:
                                rm_index_b = rand_rm(0,per_25_b,rm_index_b,seed=seed,args=args,tokens = tokens_b)
                            elif idx >= per_25_b and idx < per_50_b:
                                rm_index_b = rand_rm(per_25_b,per_50_b,rm_index_b,seed=seed,args=args,tokens = tokens_b)
                            elif idx >= per_50_b and idx < per_75_b:
                                rm_index_b = rand_rm(per_50_b,per_75_b,rm_index_b,seed=seed,args=args,tokens = tokens_b)
                            else:
                                rm_index_b = rand_rm(per_75_b,len(tokens_b),rm_index_b,seed=seed,args=args,tokens = tokens_b)

            rand_index_a.extend(rm_index_a)
            rand_index_b.extend(rm_index_b)
            tmp_ex = InputExample(guid=guid, text_a=example.text_a, text_b=example.text_b, label=label,
                                 rand_index_a=rand_index_a, rand_index_b=rand_index_b)
            auged_examples.append(tmp_ex)

            if example.text_b:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                if do_roberta:
                    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 4)
                else:
                    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3 )

                rm_tokens_b = ["[MASK]" if i in rm_index_b else x for i, x in enumerate(tokens_b)]

            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2 )]

            rm_tokens_a = ["[MASK]" if i in rm_index_a else x for i, x in enumerate(tokens_a)]

            if do_roberta:
                tokens = ["<s>"] + rm_tokens_a + ["</s>"]
            else:
                tokens = ["[CLS]"] + rm_tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            token_real_label = [1] * len(tokens)
            token_real_label[0] = -1
            token_real_label[-1] = -1

            if example.text_b:
                more_special = 0
                if do_roberta:
                    tokens += ["</s>"] + rm_tokens_b + ["</s>"]
                    more_special = 1
                else:
                    tokens += rm_tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(rm_tokens_b) + 1 + more_special)
                token_real_label += [1] * (len(rm_tokens_b) + 1 + more_special)
                token_real_label[-1] = -1

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            ignore_padding = [-1] * (max_seq_length - len(input_ids))
            input_ids += [pad_token] * (max_seq_length - len(input_ids))
            input_mask += padding
            segment_ids += padding
            token_real_label += ignore_padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(token_real_label) == max_seq_length

            if output_mode == "classification":
                label_id = label_map[example.label]
            elif output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(output_mode)

            if ex_index < num_show:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("sequence_label: %s (id = %d)" % (example.label, label_id))
                logger.info("rm_index_a: %s" % " ".join([str(x) for x in rm_index_a]))
                logger.info("rm_index_b: %s" % " ".join([str(x) for x in rm_index_b]))


            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              token_real_label=token_real_label))
            ex_index += 1

    return features, auged_examples


def Aug_each_ckpt(ori_examples, label_list, model, tokenizer, args=None, num_show=5, output_mode="classification",
                    aug_ratio=0.2, seed=42, use_bert=True, do_roberta=0, ssa_roberta=0, pad_token=0):
    if aug_ratio == 0:
        return ori_examples
    np.random.seed(seed)
    random.seed(seed)
    max_seq_length = args.max_seq_length
    all_dict = []
    len_aug = int(len(ori_examples)*aug_ratio)
    len_ori = int(len(ori_examples)-len_aug)
    cnt = 2
    num_no_aug = 1
    total_auged_ex = []
    total_k = 0
    logger.info("len_aug=%d", len_aug)

    label_map = {i: label for i, label in enumerate(label_list)}

    while len(total_auged_ex) < len_aug:
        total_k += 1
        not_qualify_index = []
        features, auged_examples = aug_func(ori_examples, label_list, tokenizer, epochs, num_no_aug, args, seed,
                                            max_seq_length, output_mode, num_show, pad_token=pad_token,
                                            do_roberta=do_roberta)
        eval_features = features
        logger.info("***** Running data aug *****")
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
        idx=0
        pred=0
        num_q=0
        num_effect=0
        pro_pred=0.0
        num_change=0
        ori_k = 0
        for batch in eval_dataloader:
            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, label_ids, token_real_label = batch

            with torch.no_grad():
                if use_bert:
                    seq_logits = model(input_ids, segment_ids, input_mask, labels=None)
                elif do_roberta:
                    if ssa_roberta:
                        seq_logits, aug_logits, aug_loss = model(input_ids, input_mask, labels=None,
                                                             token_real_label=token_real_label)
                    else:
                        outputs = model(input_ids, input_mask)
                        seq_logits = outputs[0]
                else:
                    seq_logits, aug_logits, aug_loss = model(input_ids, segment_ids, input_mask, labels=None,
                                                             token_real_label=token_real_label)

                logits = F.softmax(seq_logits, dim=-1)
                logits = logits.detach().cpu().numpy()

                seq_logits = seq_logits.detach().cpu().numpy()
                label_ids = label_ids.detach().cpu().numpy()

                if output_mode == "classification":
                    outputs = np.argmax(seq_logits, axis=1)
                    for i in range(outputs.shape[0]):
                        if idx % cnt == 0 :
                            pred=outputs[i]
                            not_qualify_index.append(idx)
                        else:
                            if pred != label_ids[i]:
                                not_qualify_index.append(idx)
                            elif outputs[i] != pred:
                                not_qualify_index.append(idx)
                            else:
                                num_q += 1

                        idx += 1
                else:
                    outputs = np.squeeze(seq_logits)
                    for i in range(outputs.shape[0]):
                        if idx % cnt == 0:
                            pred = outputs[i]
                            not_qualify_index.append(idx)
                        else:
                            #print(label_ids[i],pred,outputs[i])
                            if abs(label_ids[i] - pred) > 0.3:
                                not_qualify_index.append(idx)
                            elif abs(outputs[i] - pred) > 0.3:
                                not_qualify_index.append(idx)
                            else:
                                num_q += 1

                        idx += 1

        logger.info("total_k=%d", total_k)
        logger.info("num_of_all_examples=%d", idx)
        logger.info("num_of_ori_examples=%d", len(ori_examples))
        logger.info("not_qualify_index=%d", len(not_qualify_index))
        logger.info("qualify=%d,num_effect=%d,num_change=%d", num_q, num_effect, num_change)
        res_examples = [x for i, x in enumerate(auged_examples) if i not in not_qualify_index]
        logger.info("res_examples=%d", len(res_examples))
        real_add = 0
        for i, x in enumerate(res_examples):
            if x.__dict__ not in all_dict:
                all_dict.append(x.__dict__)
                total_auged_ex.append(x)
                real_add += 1
        logger.info("after remove same=%d,need len_aug=%d,real_add=%d", len(total_auged_ex), len_aug, real_add)


    final_examples = random.sample(total_auged_ex, len_aug) + random.sample(ori_examples, len_ori)
    if args.double_ori == 1:
        final_examples += ori_examples
    logger.info("final_examples=%d", len(final_examples))

    return final_examples
