import torch
import os
import json
import logging
import csv
import itertools
from torch.utils.data import Dataset
import random
import pdb
import argparse
from torch.utils.data import TensorDataset, random_split
from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import random
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import time
import datetime
import pickle
import copy
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

logger = logging.getLogger()

def add_reverse_relation():
    # add reverse relation
    event_pairs = json.load(open('event_triple_pairs.json','r'))
    n_event_pairs = []
    for pair in event_pairs:
        he, r, te = pair.split('\t')
        rr = '_'+r
        n_pair = [te, rr, he]
        n_event_pairs.append(pair)
        n_event_pairs.append('\t'.join(n_pair))
    random.shuffle(n_event_pairs)
    json.dump(n_event_pairs, open('double_event_triple_pairs.json', 'w'), indent=4)



import random
import math
import linecache
def split_event_triples():
    data = json.load(open('double_event_triple_pairs.json', 'r'))

    numb = len(data)
    idx = [i for i in range(numb)]
    tst_dev_idx = random.sample(idx, math.ceil(numb * 0.1))  # 默认不重复抽样（不放回的抽样）
    tst_dev_num = len(tst_dev_idx)
    tst_idx = random.sample(tst_dev_idx, math.ceil(tst_dev_num * 0.5))

    # test file
    test = open('event_pairs_test.txt', 'w', encoding='utf-8')
    dev = open('event_pairs_dev.txt', 'w', encoding='utf-8')
    train = open('event_pairs_train.txt', 'w', encoding='utf-8')

    tc = 0
    for tst_i in tst_idx:
        test.write(data[tst_i] + '\n')
        tc += 1
    print('finish test: {}'.format(tc))
    test.close()

    d_c = 0
    for dev_i in tst_dev_idx:
        if dev_i not in tst_idx:
            dev.write(data[dev_i]+'\n')
            d_c+=1
    print('finish dev: {}'.format(d_c))
    dev.close()

    tr_c = 0
    for tra_i in idx:
        if tra_i not in tst_dev_idx:
            train.write(data[tra_i] + '\n')
            tr_c += 1
    print('finish atomic train: {}'.format(tr_c))
    train.close()


def simply_event_triples():  # todo: 其实应该做个词性还原的，先试试效果吧
    # test file
    test = open('event_pairs_test.txt', 'r', encoding='utf-8')
    dev = open('event_pairs_dev.txt', 'r', encoding='utf-8')
    train = open('event_pairs_train.txt', 'r', encoding='utf-8')

    sim_test = open('sim_event_pairs_test.txt', 'w', encoding='utf-8')
    sim_dev = open('sim_event_pairs_dev.txt', 'w', encoding='utf-8')
    sim_train = open('sim_event_pairs_train.txt', 'w', encoding='utf-8')


    for line in test.readlines():
        he, r, te = line.strip('\n').split('\t')
        he = he.replace('[name]\'s', 'his')  # 类似于去掉PersonX, PersonY
        te = te.replace('[name]\'s', 'his')

        he = he.replace('[name]', '')
        te = te.replace('[name]', '')

        sim_test.write('\t'.join([he, r, te])+'\n')


    for line in dev.readlines():
        he, r, te = line.strip('\n').split('\t')
        he = he.replace('[name]\'s', 'his')  # 类似于去掉PersonX, PersonY
        te = te.replace('[name]\'s', 'his')

        he = he.replace('[name]', '')
        te = te.replace('[name]', '')

        sim_dev.write('\t'.join([he, r, te])+'\n')

    for line in train.readlines():
        he, r, te = line.strip('\n').split('\t')
        he = he.replace('[name]\'s', 'his')  # 类似于去掉PersonX, PersonY
        te = te.replace('[name]\'s', 'his')

        he = he.replace('[name]', '')
        te = te.replace('[name]', '')

        sim_train.write('\t'.join([he, r, te])+'\n')


def normalize_case(text):
    if len(text) > 1:
        try:
            normalized = text[0].upper() + text[1:].lower()
            if normalized[-1] != '.':
                normalized = normalized + '.'
        except:
            raise RuntimeError("Cannot normalize text {}".format(text))
        return normalized
    return text

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


class DatasetHelper(Dataset):
    def __init__(self, data_path, knowledge_path=None, src_max_length=100, tgt_max_length=100,
                 do_generate=False):
        self.do_generate = do_generate
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length
        # self.tokenizer = tokenizer
        # self.bos = self.tokenizer.encoder["<|bos|>"]
        # self.pad = self.tokenizer.encoder["<|pad|>"]
        # self.eos = self.tokenizer.encoder["<|endoftext|>"]
        # self.ctx = self.tokenizer.encoder["<|ctx|>"]
        # self.kid = self.tokenizer.encoder["<|kg|>"]
        self.data_path = data_path
        self.knowledge_path = knowledge_path  # for downstream task.

        self.r2id = {'xAttr':0,'xEffect':1,'xIntent':2,'xNeed':3,'xReact':4,'xWant':5,
                     'oEffect':6,'oReact':7,'oWant':8, '_xAttr':9,'_xEffect':10,'_xIntent':11,
                     '_xNeed':12,'_xReact':13,'_xWant':14, '_oEffect':15,'_oReact':16,'_oWant':17}
        self.id2r = {0: 'xAttr', 1: 'xEffect', 2: 'xIntent', 3: 'xNeed', 4: 'xReact', 5: 'xWant',
                     6: 'oEffect', 7: 'oReact', 8: 'oWant', 9: '_xAttr', 10: '_xEffect', 11: '_xIntent',
                     12: '_xNeed', 13: '_xReact', 14: '_xWant', 15: '_oEffect', 16: '_oReact', 17: '_oWant'}
    def load(self):
        self.source = []
        self.attn_mask = []
        self.target = []
        with open(self.data_path, 'r') as f:
            for line in f.readlines():
                triples = line.strip('\n').split('\t')
                head_event = triples[0]
                tail_event = triples[2]
                relation = triples[1]

                encoded_dict = tokenizer.encode_plus(text=head_event,
                                                     text_pair=tail_event,
                                                     add_special_tokens=True,
                                                     max_length=50,
                                                     pad_to_max_length=True,
                                                     return_attention_mask=True,
                                                     return_tensors='pt')
                self.source.append(encoded_dict['input_ids'])
                self.attn_mask.append(encoded_dict['attention_mask'])
                self.target.append(self.r2id[relation])

        input_ids = torch.cat(self.source, dim=0)
        attention_masks = torch.cat(self.attn_mask, dim=0)
        labels = torch.tensor(self.target)

        # Print sentence 0, now as a list of IDs.
        for i in range(20,23):
            print('input: ', tokenizer.decode(input_ids[i]))
            print('attention mask:', attention_masks[i])
            print('label: ', labels[i])

        return input_ids, attention_masks, labels


    def load_downstream_dataset(self):
        self.source = []
        self.attn_mask = []
        with open(self.data_path, 'r') as f:
            for line in f.readlines():
                triples = line.strip('\n').split('\t')
                head_event = triples[0]
                tail_event = triples[1]
                encoded_dict = tokenizer.encode_plus(text=head_event,
                                                     text_pair=tail_event,
                                                     add_special_tokens=True,
                                                     max_length=50,
                                                     pad_to_max_length=True,
                                                     return_attention_mask=True,
                                                     return_tensors='pt')
                self.source.append(encoded_dict['input_ids'])
                self.attn_mask.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(self.source, dim=0)
        attention_masks = torch.cat(self.attn_mask, dim=0)

        # Print sentence 0, now as a list of IDs.
        for i in range(20, 23):
            print('input: ', tokenizer.decode(input_ids[i]))
            print('attention mask:', attention_masks[i])

        return input_ids, attention_masks


    def __len__(self):
        return len(self.source)



def define_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file", default="/apdcephfs/share_916081/qtli/kggen/data/knowledge/atomic/event_pairs/sim_event_pairs_train.txt", type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--dev_data_file", default="/apdcephfs/share_916081/qtli/kggen/data/knowledge/atomic/event_pairs/sim_event_pairs_dev.txt", type=str,
                        help="The input validation data file (a text file).")
    parser.add_argument("--test_data_file", default="/apdcephfs/share_916081/qtli/kggen/data/knowledge/atomic/event_pairs/sim_event_pairs_test.txt", type=str,
                        help="The input testing data file (a text file).")
    parser.add_argument("--output_dir", default="/apdcephfs/share_916081/qtli/kggen/code/event_relation_classifier/trained_bert_output/", type=str, help="output dir saving trained models.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument("--workers", default=7, type=int, help="workers")
    parser.add_argument("--num_labels", default=18, type=int, help="workers")
    parser.add_argument("--epochs", default=5, type=int, help="epochs")
    parser.add_argument("--warmup_ratio", default=0.1, type=float, help="Linear warmup over warmup_ratio.")
    parser.add_argument("--mode", default="train", type=str, help="output dir saving trained models.")

    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--logging_steps', type=int, default=100, help="Log every X updates steps.")
    parser.add_argument('--validate_steps', type=int, default=1000, help="evaluate model every x updates steps")
    parser.add_argument("--plot", action='store_true', help="plot table/curve")
    parser.add_argument("--continue_train", action='store_true', help="plot table/curve")

    parser.add_argument("--prediction_output",
                        default="pred.pkl",
                        type=str,
                        help="The predictions for downstream dataset.")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    return args

def train(args):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataset = DatasetHelper(data_path=args.train_data_file, do_generate=False)
    input_ids, attention_masks, labels = train_dataset.load()
    train_dataset = TensorDataset(input_ids, attention_masks, labels)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  drop_last=False,
                                  num_workers=args.workers,
                                  pin_memory=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    val_dataset = DatasetHelper(data_path=args.dev_data_file, do_generate=False)
    input_ids, attention_masks, labels = val_dataset.load()
    val_dataset = TensorDataset(input_ids, attention_masks, labels)

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=args.eval_batch_size,  # Evaluate with this batch size.
        num_workers=args.workers,
        pin_memory=True,
    )
    if not args.continue_train:
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=args.num_labels,  # The number of output labels--2 for binary classification.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
    else:
        model = BertForSequenceClassification.from_pretrained(
            args.output_dir,  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=args.num_labels,  # The number of output labels--2 for binary classification.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
    # Tell pytorch to run this model on the GPU.
    model.to(args.device)

    # Don't apply weight decay to any parameters whose names include these tokens.
    # (Here, the BERT doesn't have `gamma` or `beta` parameters, only `bias` terms)
    # tells the optimizer to not apply weight decay to the bias terms (e.g., $ b $ in the equation $ y = Wx + b $ ).
    # Weight decay is a form of regularization–after calculating the gradients, we multiply them by, e.g., 0.99.
    no_decay = ['bias', 'LayerNorm.weight']
    # Separate the `weight` parameters from the `bias` parameters.
    # - For the `weight` parameters, this specifies a 'weight_decay_rate' of 0.01.
    # - For the `bias` parameters, the 'weight_decay_rate' is 0.0.
    optimizer_grouped_parameters = [
        # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.1},

        # Filter for parameters which *do* include those.
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    # optimizer = AdamW(model.parameters(),
    #                   lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
    #                   eps=1e-8  # args.adam_epsilon  - default is 1e-8.
    #                   )
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )

    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = args.epochs

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Don't apply weight decay to any parameters whose names include these tokens.
    # (Here, the BERT doesn't have `gamma` or `beta` parameters, only `bias` terms)

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(args.warmup_ratio * total_steps),
                                                # Default value in run_glue.py
                                                num_training_steps=total_steps)

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()
    global_step = 0
    # device = 'cuda'
    last_accuracy = 0.0
    last_loss = 100.0
    patient = 0

    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("Saving model to %s" % args.output_dir)

    # For each epoch...
    for epoch_i in range(0, epochs):
        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            global_step += 1
            # Progress update every 40 batches.
            if global_step % args.logging_steps == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(args.device)
            b_input_mask = batch[1].to(args.device)
            b_labels = batch[2].to(args.device)
            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            train_outputs = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += train_outputs[0].item()

            # Perform a backward pass to calculate the gradients.
            train_outputs[0].backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

            if global_step % args.validate_steps == 0 and step != 0:
                # ========================================
                #               Validation
                # ========================================
                # After the completion of each training epoch, measure our performance on
                # our validation set.

                print("")
                print("Running Validation...")

                t0 = time.time()

                # Put the model in evaluation mode--the dropout layers behave differently
                # during evaluation.
                model.eval()

                # Tracking variables
                total_eval_accuracy = 0
                total_eval_loss = 0
                nb_eval_steps = 0
                # Evaluate data for one epoch
                for batch in validation_dataloader:
                    # Unpack this training batch from our dataloader.
                    #
                    # As we unpack the batch, we'll also copy each tensor to the GPU using
                    # the `to` method.
                    #
                    # `batch` contains three pytorch tensors:
                    #   [0]: input ids
                    #   [1]: attention masks
                    #   [2]: labels
                    b_input_ids = batch[0].to(args.device)
                    b_input_mask = batch[1].to(args.device)
                    b_labels = batch[2].to(args.device)

                    # Tell pytorch not to bother with constructing the compute graph during
                    # the forward pass, since this is only needed for backprop (training).
                    with torch.no_grad():
                        # Forward pass, calculate logit predictions.
                        # token_type_ids is the same as the "segment ids", which
                        # differentiates sentence 1 and 2 in 2-sentence tasks.
                        # The documentation for this `model` function is here:
                        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                        # Get the "logits" output by the model. The "logits" are the output
                        # values prior to applying an activation function like the softmax.
                        val_outputs = model(b_input_ids,
                                               token_type_ids=None,
                                               attention_mask=b_input_mask,
                                               labels=b_labels)
                    # Accumulate the validation loss.
                    total_eval_loss += val_outputs[0].item()

                    # Move logits and labels to CPU
                    logits = val_outputs[1].detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()

                    # Calculate the accuracy for this batch of test sentences, and
                    # accumulate it over all batches.
                    total_eval_accuracy += flat_accuracy(logits, label_ids)

                # Report the final accuracy for this validation run.
                avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
                print("Accuracy: {0:.2f}".format(avg_val_accuracy))

                # Calculate the average loss over all of the batches.
                avg_val_loss = total_eval_loss / len(validation_dataloader)

                # Measure how long the validation run took.
                validation_time = format_time(time.time() - t0)

                print("Validation Loss: {0:.2f}".format(avg_val_loss))
                print("Validation took: {:}".format(validation_time))

                # Record all statistics from this epoch.
                training_stats.append(
                    {
                        'epoch': epoch_i + 1,
                        'global_step': global_step,
                        'Valid. Loss': avg_val_loss,
                        'Valid. Accur.': avg_val_accuracy,
                        'Validation Time': validation_time
                    }
                )
                # Validation Loss is a more precise measure than accuracy,
                # because with accuracy we don’t care about the exact output value, but just which side of a threshold it falls on.
                if avg_val_loss <= last_loss:
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                    # Good practice: save your training arguments together with the trained model
                    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

                    last_loss = avg_val_loss
                else:
                    patient += 1
                    print('patient is {}.'.format(patient))
                if patient > 3: break
                model.train()
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    if args.plot:
        # Display floats with two decimal places.
        pd.set_option('precision', 2)

        # Create a DataFrame from our training statistics.
        df_stats = pd.DataFrame(data=training_stats)

        # Use the 'epoch' as the row index.
        df_stats = df_stats.set_index('epoch')

        # A hack to force the column headers to wrap.
        # df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

        # Display the table.
        print(df_stats)
        show_loss_curve(df_stats)

import matplotlib.pyplot as plt
import seaborn as sns

def show_loss_curve(df_stats):
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])

    plt.show()


def test(args):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    test_dataset = DatasetHelper(data_path=args.test_data_file, do_generate=False)
    input_ids, attention_masks, labels = test_dataset.load()
    test_dataset = TensorDataset(input_ids, attention_masks, labels)

    # For validation the order doesn't matter, so we'll just read them sequentially.
    test_dataloader = DataLoader(
                                test_dataset,  # The validation samples.
                                sampler=SequentialSampler(test_dataset),  # Pull out batches sequentially.
                                batch_size=args.eval_batch_size,  # Evaluate with this batch size.
                                num_workers=args.workers,
                                pin_memory=True,)

    model = BertForSequenceClassification.from_pretrained(
        args.output_dir,  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=args.num_labels,  # The number of output labels--2 for binary classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model.to(args.device)
    model.eval()
    predictions, true_labels = [], []
    total_eval_accuracy = 0
    # Predict
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(args.device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    print("Accuracy: {0:.2f}".format(avg_val_accuracy))

from sklearn.metrics import matthews_corrcoef
def cal_mcc(predictions, true_labels):
    matthews_set = []

    # Evaluate each test batch using Matthew's correlation coefficient
    print('Calculating Matthews Corr. Coef. for each batch...')

    # For each input batch...
    for i in range(len(true_labels)):
        # The predictions for this batch are a 2-column ndarray (one column for "0"
        # and one column for "1"). Pick the label with the highest value and turn this
        # in to a list of 0s and 1s.
        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()

        # Calculate and store the coef for this batch.
        matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
        matthews_set.append(matthews)

    # Create a barplot showing the MCC score for each batch of test samples.
    ax = sns.barplot(x=list(range(len(matthews_set))), y=matthews_set, ci=None)

    plt.title('MCC Score per Batch')
    plt.ylabel('MCC Score (-1 to +1)')
    plt.xlabel('Batch #')

    plt.show()

    # Combine the results across all batches.
    flat_predictions = np.concatenate(predictions, axis=0)

    # For each sample, pick the label (0 or 1) with the higher score.
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)

    # Calculate the MCC
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

    print('Total MCC: %.3f' % mcc)


class JsonDumpHelper(json.JSONEncoder):
    def default(self, obj):
        if type(obj) != str:
            return str(obj)
        return json.JSONEncoder.default(self, obj)

def infer(args):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    test_dataset = DatasetHelper(data_path=args.train_data_file, do_generate=False)
    input_ids, attention_masks = test_dataset.load_downstream_dataset()
    test_dataset = TensorDataset(input_ids, attention_masks)

    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), cls=JsonDumpHelper, indent=4, sort_keys=True))
    logger.info('-' * 100)

    # For validation the order doesn't matter, so we'll just read them sequentially.
    test_dataloader = DataLoader(
                                test_dataset,  # The validation samples.
                                sampler=SequentialSampler(test_dataset),  # Pull out batches sequentially.
                                batch_size=args.eval_batch_size,  # Evaluate with this batch size.
                                num_workers=args.workers,
                                pin_memory=True,)

    model = BertForSequenceClassification.from_pretrained(
        args.output_dir,  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=args.num_labels,  # The number of output labels--2 for binary classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model.to(args.device)
    model.eval()
    predictions, true_labels = [], []
    total_eval_accuracy = 0
    # Predict
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(args.device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()

        # Store predictions and true labels
        # print('pred_flat')
        # pdb.set_trace()
        predictions.extend(pred_flat.tolist())

    pickle.dump(predictions, open(args.prediction_output, 'wb'))


id2r = {0: 'xAttr', 1: 'xEffect', 2: 'xIntent', 3: 'xNeed', 4: 'xReact', 5: 'xWant',
         6: 'oEffect', 7: 'oReact', 8: 'oWant', 9: '_xAttr', 10: '_xEffect', 11: '_xIntent',
         12: '_xNeed', 13: '_xReact', 14: '_xWant', 15: '_oEffect', 16: '_oReact', 17: '_oWant'}
import copy
# def feed_pred_relation_story():
#     args = define_args()
#
#     pred = pickle.load(open(args.prediction_output,'rb'))
#     # data = json.load(open(args.train_data_file,'r'))
#     data = json.load(open(args.train_data_file,'r'))
#     new_data = []
#     idx = 0
#     for item in data:
#         event = item['story_event']
#         tmp = []
#         for i, es in enumerate(event):
#             tmp_es = []
#             for j, e in enumerate(es):
#                 if i == len(event)-1 and j==len(es)-1:
#                     tmp_es.append(e)
#                 else:
#                     tmp_es.append(e)
#                     tmp_es.append(id2r[int(pred[idx])])
#                     idx +=1
#             tmp.append(tmp_es)
#         item['story_events'] = tmp
#         new_data.append(item)
#     # json.dump(new_data, open('/mnt/lustre/kennethkong/qtdir/gen_w_kg/data/story_ending_generation/ROCstory/dev_new.json','w'), indent=4)
#     json.dump(new_data, open(args.train_data_file, 'w'), indent=4)


def feed_pred_relation_story(args):
    pred = pickle.load(open(args.prediction_output,'rb'))
    # data = json.load(open(args.train_data_file,'r'))
    data = json.load(open(args.train_data_file,'r'))
    new_data = []
    idx = 0
    for item in data:
        event = item['story_sim_events']
        tmp = []
        for i, es in enumerate(event):  # 遍历每个utterance的events
            tmp_es = []  # 包含这个utterance涉及的events和relations
            for j, e in enumerate(es):  # 遍历这个utterance的所有events
                if i==0 and j==0:  # 如果是第一个utterance的第一个event
                    tmp_es.append(e)
                else:
                    tmp_es.append(id2r[int(pred[idx])])
                    tmp_es.append(e)
                    idx +=1
            tmp.append(tmp_es)
        item['story_events_relations'] = tmp
        new_data.append(item)
    json.dump(new_data, open(args.train_data_file, 'w'), indent=4)


def feed_pred_relation_dialogue(args):
    pred = pickle.load(open(args.prediction_output,'rb'))
    # data = json.load(open(args.train_data_file,'r'))
    data = json.load(open(args.train_data_file,'r'))
    new_data = []
    idx = 0
    # 关于预测的 relation 插在哪里？
    '''
    "dialogue_events": [
            [
                "my friend leave",
                "she be housesitt for i",
                "my cat snuck out",
                "my cat be kill"
            ],
            [
                "that isnt good be mad at friend"
            ],
            [
                "once i have some time",
                "i realize",
                "she feel really bad about it"
            ],
            [
                "be you go to get another cat"
            ]
        ]
    对于第1个utterance，relation是插在两个event之间
    对于第2个utterance到最后一句（response），relation插在每个event的前面，每个utterance的最后event后面不要跟relation
    '''
    for item in data:
        event = item['dialogue_events']
        tmp = []
        for i, es in enumerate(event):  # 遍历每个utterance的events
            tmp_es = []  # 包含这个utterance涉及的events和relations
            for j, e in enumerate(es):  # 遍历这个utterance的所有events
                if i==0 and j==0:  # 如果是第一个utterance的第一个event
                    tmp_es.append(e)
                else:
                    tmp_es.append(id2r[int(pred[idx])])
                    tmp_es.append(e)
                    idx +=1
                # if i == len(event)-1 and j==len(es)-1:
                #     tmp_es.append(e)
                # else:
                #     tmp_es.append(e)
                #     tmp_es.append(id2r[int(pred[idx])])
                #     idx +=1
            tmp.append(tmp_es)

        # print('check the chain of each item')
        # pdb.set_trace()
        item['dialogue_events_relations'] = tmp
        new_data.append(item)
    json.dump(new_data, open(args.train_data_file, 'w'), indent=4)


if __name__ == '__main__':
    args = define_args()

    if 'train' in args.mode:
        train(args)
    if 'test' in args.mode:
        test(args)
    infer(args)
    # feed_pred_relation_story()
    # feed_pred_relation_dialogue()
