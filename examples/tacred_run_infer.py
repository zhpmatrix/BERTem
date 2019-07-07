from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random

import sys
sys.path.append('..')
os.environ['CUDA_VISIBLE_DEVICES']='0'

import copy
import json
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, classification_report,precision_recall_fscore_support

from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer

from pprint import pprint
import textdistance
import neuralcoref
import en_core_web_sm
from itertools import groupby, combinations
from utils import get_candidate_input

from flask import Flask,request,jsonify
app = Flask(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, entity_pos=None):
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
        self.entity_pos = entity_pos

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,input_ids, input_mask, segment_ids, label_id, entity_mask=None, entity_seg_pos=None, entity_span1_pos=None, entity_span2_pos=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.entity_mask = entity_mask
        self.entity_seg_pos = entity_seg_pos
        self.entity_span1_pos = entity_span1_pos
        self.entity_span2_pos = entity_span2_pos


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class TacredProcessor(DataProcessor):
    """Processor for the TACRED dataset."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.jsonl")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_dev.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.jsonl")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ['per:parents', 'per:country_of_birth', 'org:political/religious_affiliation', 'org:parents', 'org:members', 'per:schools_attended', 'org:shareholders', 'per:stateorprovince_of_death', 'per:age', 'per:city_of_death', 'per:siblings', 'per:date_of_birth', 'org:founded', 'per:stateorprovince_of_birth', 'per:origin', 'per:charges', 'per:children', 'per:title', 'per:countries_of_residence', 'org:top_members/employees', 'per:religion', 'per:country_of_death', 'per:employee_of', 'no_relation', 'per:stateorprovinces_of_residence', 'org:city_of_headquarters', 'org:dissolved', 'per:date_of_death', 'per:other_family', 'per:alternate_names', 'org:number_of_employees/members', 'per:spouse', 'per:cause_of_death', 'org:alternate_names', 'org:founded_by', 'org:stateorprovince_of_headquarters', 'per:city_of_birth', 'org:subsidiaries', 'org:website', 'org:member_of', 'per:cities_of_residence', 'org:country_of_headquarters']
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            line = json.loads(line[0])
            text_a = ' '.join(line['tokens'])
            label = line['label']
            entity_pos = line['entities']
            # 假设entity之间不重叠
            entity_pos = sorted(entity_pos) 
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label, entity_pos = entity_pos))
        return examples

class _TacredProcessor(DataProcessor):
    """Processor for the TACRED dataset."""

    def get_test_examples(self, lines):
        """See base class."""
        return self._create_examples(lines, "test")

    def get_labels(self):
        """See base class."""
        return ['per:parents', 'per:country_of_birth', 'org:political/religious_affiliation', 'org:parents', 'org:members', 'per:schools_attended', 'org:shareholders', 'per:stateorprovince_of_death', 'per:age', 'per:city_of_death', 'per:siblings', 'per:date_of_birth', 'org:founded', 'per:stateorprovince_of_birth', 'per:origin', 'per:charges', 'per:children', 'per:title', 'per:countries_of_residence', 'org:top_members/employees', 'per:religion', 'per:country_of_death', 'per:employee_of', 'no_relation', 'per:stateorprovinces_of_residence', 'org:city_of_headquarters', 'org:dissolved', 'per:date_of_death', 'per:other_family', 'per:alternate_names', 'org:number_of_employees/members', 'per:spouse', 'per:cause_of_death', 'org:alternate_names', 'org:founded_by', 'org:stateorprovince_of_headquarters', 'per:city_of_birth', 'org:subsidiaries', 'org:website', 'org:member_of', 'per:cities_of_residence', 'org:country_of_headquarters']
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(line['tokens'])
            label = line['label']
            entity_pos = line['entities']
            # 假设entity之间不重叠
            entity_pos = sorted(entity_pos) 
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label, entity_pos = entity_pos))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    reverse_label_map = {i : label for i, label in enumerate(label_list)}
    samples = []
    features = []
    for (ex_index, example) in enumerate(examples):
        old_entity_pos = copy.deepcopy(example.entity_pos)
        tokens_a, new_entity_pos = tokenizer.tokenize(example.text_a,example.entity_pos)
        
        old_entity0_ = ' '.join(example.text_a.split()[old_entity_pos[0][0]:old_entity_pos[0][1]])
        old_entity1_ = ' '.join(example.text_a.split()[old_entity_pos[1][0]:old_entity_pos[1][1]])
        
        old_entity0 = ''.join(example.text_a.split()[old_entity_pos[0][0]:old_entity_pos[0][1]])
        old_entity1 = ''.join(example.text_a.split()[old_entity_pos[1][0]:old_entity_pos[1][1]])
        new_entity0 = ''.join(tokens_a[new_entity_pos[0][0]:new_entity_pos[0][1]])
        new_entity1 = ''.join(tokens_a[new_entity_pos[1][0]:new_entity_pos[1][1]])
        
        old_entity0 = old_entity0.lower()
        old_entity1 = old_entity1.lower()
        if '##' in new_entity0 or '##' in new_entity1:
            new_entity0 = new_entity0.replace('#','')
            new_entity1 = new_entity1.replace('#','')
        
        try:
            assert(old_entity0 == new_entity0)
            assert(old_entity1 == new_entity1)
        except:
            continue
        # Entity marker
        tokens_a_ = copy.deepcopy(tokens_a) 
        new_entity_pos_ = copy.deepcopy(new_entity_pos)
        entity1_start, entity1_end = new_entity_pos[0][0], new_entity_pos[0][1] 
        entity2_start, entity2_end = new_entity_pos[1][0], new_entity_pos[1][1] 
        
        tokens_a.insert(entity1_start, '<s1>') 
        new_entity_pos[0][0] = entity1_start
        tokens_a.insert(entity1_end+1, '<e1>')
        new_entity_pos[0][1] = entity1_end+1+1
        tokens_a.insert(entity2_start+2, '<s2>')
        new_entity_pos[1][0] = entity2_start+2
        tokens_a.insert(entity2_end+3,'<e2>')
        new_entity_pos[1][1] = entity2_end+3+1

        if new_entity_pos[1][1] > max_seq_length - 2 - 1:
            continue
            
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        

        # Used for mention pooling
        entity_mask_tag = 1
        entity_mask = [0] * len(input_ids)
        for entity in new_entity_pos:
            start, end = entity[0],entity[1]
            for i in range(start, end):
                # [CLS], need to +1 offset
                entity_mask[i+1] = entity_mask_tag
        
        """
            Different position embedding
        """
        # Strategy 1
        entity1_pos_tag = 1
        entity2_pos_tag = 2

        entity_seg_pos = [0] * len(input_ids)

        entity1_start, entity1_end = new_entity_pos[0][0], new_entity_pos[0][1] 
        for i in range(entity1_start, entity1_end):
            entity_seg_pos[i+1] = entity1_pos_tag
        entity2_start, entity2_end = new_entity_pos[1][0], new_entity_pos[1][1] 
        for i in range(entity2_start, entity2_end):
            entity_seg_pos[i+1] = entity2_pos_tag
        
        # Strategy 2
        entity_start_pos_tag = 1
        entity_seg_pos_ = [0] * len(input_ids)
        entity1_start, entity1_end = new_entity_pos[0][0], new_entity_pos[0][1] 
        entity_seg_pos_[entity1_start+1] = entity_start_pos_tag
        entity2_start, entity2_end = new_entity_pos[1][0], new_entity_pos[1][1] 
        entity_seg_pos_[entity2_start+1] = entity_start_pos_tag

        # Strategy 3
        entity_span1_pos = [0] * len(input_ids)
        entity1_start, entity1_end = new_entity_pos[0][0], new_entity_pos[0][1] 
        for i in range(len(entity_span1_pos)):
            if i < entity1_start:
                #entity_span1_pos[i] = np.abs(i - entity1_start)
                entity_span1_pos[i] = i - entity1_start
            elif entity1_start <= i and i < entity1_end:
                entity_span1_pos[i] = 0
            elif i >= entity1_end:
                entity_span1_pos[i] = i - entity1_end + 1
        
        entity_span2_pos = [0] * len(input_ids)
        entity2_start, entity2_end = new_entity_pos[1][0], new_entity_pos[1][1] 
        for i in range(len(entity_span2_pos)):
            if i < entity2_start:
                #entity_span2_pos[i] = np.abs(i - entity2_start)
                entity_span2_pos[i] = i - entity2_start
            elif entity2_start <= i and i < entity2_end:
                entity_span2_pos[i] = 0
            elif i >= entity2_end:
                entity_span2_pos[i] = i - entity2_end + 1

        # Avoid to get negative position to fuck the nn.Embedding
        #entity_span1_pos = [pos+max_seq_length-1 for pos in entity_span1_pos]
        #entity_span2_pos = [pos+max_seq_length-1 for pos in entity_span2_pos]
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(entity_mask) == max_seq_length
        assert len(entity_seg_pos) == max_seq_length
        assert len(entity_seg_pos_) == max_seq_length
        assert len(entity_span1_pos) == max_seq_length
        assert len(entity_span2_pos) == max_seq_length
        if output_mode == "classification":
            label_id = label_map[example.label]
        else:
            raise KeyError(output_mode)

        print("tokens: %s" % " ".join([str(x) for x in tokens]))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        print("entity_mask: %s" % " ".join([str(x) for x in entity_mask]))
        print("entity_seg_pos: %s" % " ".join([str(x) for x in entity_seg_pos]))
        print("entity_seg_pos_: %s" % " ".join([str(x) for x in entity_seg_pos_]))
        print("entity_span1_pos: %s" % " ".join([str(x) for x in entity_span1_pos]))
        print("entity_span2_pos: %s" % " ".join([str(x) for x in entity_span2_pos]))
        print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        print("label: %s (id = %d)" % (example.label, label_id))

        samples.append([example.text_a, (old_entity0_,old_entity1_)])
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              entity_mask=entity_mask,
                              entity_seg_pos=entity_seg_pos_,
                              entity_span1_pos=entity_span1_pos,
                              entity_span2_pos=entity_span2_pos))
    return features, samples, reverse_label_map

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

def load_model():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default='/data/share/zhanghaipeng/tre/datasets/data/tacred/',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model",
                        default='model/bert-base-uncased',
                        type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, ""bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, ""bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default='tacred',
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='train/23/tacred3.0',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="cache/",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--test_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',                            type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    args = parser.parse_args()


    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1

    task_name = args.task_name.lower()
    processor = _TacredProcessor()
    output_mode = 'classification'

    label_list = processor.get_labels()
    num_labels = len(label_list)
    # Load a trained model and vocabulary that you have fine-tuned
    model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    return model, tokenizer, label_list, args, output_mode, processor, device

def get_helper_model(spacy_used=False):
    from simplex_sdk import SimplexClient
    if spacy_used:
        model = en_core_web_sm.load() 
    else:
        model = SimplexClient('BertNerApi-tmp')
    return model

# 载入模型
model, tokenizer, label_list, args, output_mode, processor, device = load_model()
ner_model = get_helper_model(spacy_used=False)

@app.route('/nre')
def predict():
    data = request.args
    if 'text' not in data.keys():
        warning_str = 'pleasen input right arg!\n'
        return jsonify(warning_str)
    else:
        line = data['text']
    
    relation = []
    candidate_input_list, entity = get_candidate_input(line, ner_model,spacy_used=False)
    # 实体未找到
    if not candidate_input_list:
        return jsonify({'relations':relation,'entities':entity})
        
    test_examples = processor.get_test_examples(candidate_input_list)
    
    test_features,samples,reverse_label_map = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    
    test_num = len(candidate_input_list)
    for i in range(test_num):
        f,sample = test_features[i],samples[i]
        input_ids = torch.tensor(f.input_ids,dtype=torch.long).reshape(1, args.max_seq_length).to(device)
        segment_ids = torch.tensor(f.segment_ids,dtype=torch.long).reshape(1, args.max_seq_length).to(device)
        input_mask = torch.tensor(f.input_mask,dtype=torch.long).reshape(1, args.max_seq_length).to(device)
        entity_mask = torch.tensor(f.entity_mask,dtype=torch.float).reshape(1, args.max_seq_length).to(device)
        entity_seg_pos = torch.tensor(f.entity_seg_pos,dtype=torch.long).reshape(1, args.max_seq_length).to(device)
        entity_span1_pos = torch.tensor(f.entity_span1_pos,dtype=torch.float).reshape(1, args.max_seq_length).to(device)
        entity_span2_pos = torch.tensor(f.entity_span2_pos,dtype=torch.float).reshape(1, args.max_seq_length).to(device)
        
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, entity_mask, entity_seg_pos, entity_span1_pos, entity_span2_pos, labels=None)
        """
            np.argmax v.s. torch.argmax
        """
        pred_id = np.argmax(logits.detach().cpu().numpy()[0].tolist()) 
        pred_label = reverse_label_map[pred_id]
        text,entity0, entity1 = sample[0], sample[1][0], sample[1][1]
        relation.append({'text':text,'entity pair':[entity0,entity1],'relation':pred_label})
    return jsonify({'relations':relation,'entities':entity})

if __name__ == "__main__":
    app.run(host='0.0.0.0',port='5050',debug=True)
