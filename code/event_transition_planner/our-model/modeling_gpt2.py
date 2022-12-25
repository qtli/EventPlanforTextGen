# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
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
"""PyTorch OpenAI GPT-2 model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import json
import logging
import math
import os
import sys
from io import open
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.parameter import Parameter

# from torch_scatter import scatter_max, scatter_mean, scatter_add

from transformers.modeling_utils import PreTrainedModel, Conv1D, prune_conv1d_layer, SequenceSummary
from transformers import GPT2Config
from transformers.file_utils import add_start_docstrings
from transformers import BertModel, BertConfig

import numpy as np

logger = logging.getLogger(__name__)

GPT2_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "gpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin",
    "gpt2-medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-pytorch_model.bin",
    "gpt2-large": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-pytorch_model.bin",
    "distilgpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/distilgpt2-pytorch_model.bin", }


def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
                     "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split('/')
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+\d+', m_name):
                l = re.split(r'(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'w' or l[0] == 'g':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'b':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'wpe' or l[0] == 'wte':
                pointer = getattr(pointer, l[0])
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        self.output_attentions = config.output_attentions

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd:ns, :ns].clone()
        if head_mask is not None:
            b *= head_mask
            b += (head_mask == 0).float()
        w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class KG_Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(KG_Attention, self).__init__()
        self.output_attentions = config.output_attentions

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state  # 768
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

        self.k_attn = Conv1D(n_state * 2, nx)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, copy=False):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))  # torch.Size([5, 12, 110, 110])
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd:ns, :ns].clone()  # self.bias: torch.Size([1, 1, 1024, 1024])
        '''
        b: torch.Size([1, 1, 110, 110])  decoder's feel
        tensor([[[[1., 0., 0.,  ..., 0., 0., 0.],
          [1., 1., 0.,  ..., 0., 0., 0.],
          [1., 1., 1.,  ..., 0., 0., 0.],
          ...,
          [1., 1., 1.,  ..., 1., 0., 0.],
          [1., 1., 1.,  ..., 1., 1., 0.],
          [1., 1., 1.,  ..., 1., 1., 1.]]]])
        '''
        if head_mask is not None:
            b *= head_mask  # 清空
            b += (head_mask == 0).float()
            '''
            context 是可以看见的，所有样本普适
            tensor([[[[1., 1., 1.,  ..., 0., 0., 0.],
              [1., 1., 1.,  ..., 0., 0., 0.],
              [1., 1., 1.,  ..., 0., 0., 0.],
              ...,
              [1., 1., 1.,  ..., 1., 0., 0.],
              [1., 1., 1.,  ..., 1., 1., 0.],
              [1., 1., 1.,  ..., 1., 1., 1.]]]])
            '''
        w = w * b - 1e4 * (1 - b)  # 使用对角矩阵对attention weight进行mask操作
        if attention_mask is not None:  # torch.Size([5, 1, 1, 110])
            # Apply the attention mask
            w = w + attention_mask  # 把context里的PAD部分隐藏

        w = nn.Softmax(dim=-1)(w)  # 得到0-1的权重
        w = self.attn_dropout(w)

        outputs = [torch.matmul(w, v)]  # torch.matmul(w, v)是一个head的context向量
        if self.output_attentions or copy:
            outputs.append(w)
        return outputs

    def _kg_attn(self, q, k, v, pred_kg_attention_mask=None, copy=False, ):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))  # torch.Size([5, 12, 110, 110])

        if pred_kg_attention_mask is not None:  # torch.Size([5, 1, 1, 110])
            # Apply the attention mask
            w = w + pred_kg_attention_mask  # 把context里的PAD部分隐藏

        w = nn.Softmax(dim=-1)(w)  # 得到0-1的权重
        w = self.attn_dropout(w)

        outputs = [torch.matmul(w, v)]  # torch.matmul(w, v)是一个head的context向量
        if self.output_attentions or copy:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, pred_kg=None,
                pred_kg_attention_mask=None, copy=False, ):
        x = self.c_attn(x)  # torch.Size([5, 110, 2304])  768*3=2304
        query, key, value = x.split(self.split_size, dim=2)  # torch.Size([5, 110, 768])
        query = self.split_heads(query)  # torch.Size([5, 110, 768])--> torch.Size([5, 12, 110, 64])
        key = self.split_heads(key, k=True)  # torch.Size([5, 110, 768]) --> torch.Size([5, 12, 64, 110])
        value = self.split_heads(value)  # torch.Size([5, 110, 768]) --> torch.Size([5, 12, 110, 64])
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1),
                               value))  # transpose to have same shapes for stacking  torch.Size([2, 5, 12, 110, 64])
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, copy=copy, )
        # attn_outputs 第一项是 context vector, 第二项是 attention weight

        a = attn_outputs[0]  # torch.Size([5, 12, 110, 64])
        a = self.merge_heads(a)  # (bsz 5, seq_len 110, 768)
        a = self.c_proj(a)
        a = self.resid_dropout(a)  # torch.Size([5, 110, 768])
        outputs = [a, present] + attn_outputs[1:]

        # todo: knowledge attention is here
        if pred_kg is not None:
            k = self.k_attn(pred_kg)  # torch.Size([5, 110, 2304])  768*3=2304
            k_key, k_value = k.split(self.split_size, dim=2)
            k_key = self.split_heads(k_key, k=True)  # torch.Size([5, 110, 768]) --> torch.Size([5, 12, 64, 110])
            k_value = self.split_heads(k_value)  # torch.Size([5, 110, 768]) --> torch.Size([5, 12, 110, 64])
            k_present = torch.stack((k_key.transpose(-2, -1),
                                     k_value))  # transpose to have same shapes for stacking  torch.Size([2, 5, 12, 110, 64])
            k_attn_outputs = self._kg_attn(query, k_key, k_value, pred_kg_attention_mask=pred_kg_attention_mask,
                                           copy=copy, )
            ka = k_attn_outputs[0]
            ka = self.merge_heads(ka)
            ka = self.c_proj(ka)
            ka = self.resid_dropout(ka)  # torch.Size([5, 110, 768])
            k_outputs = [ka, k_present] + k_attn_outputs[1:]
        else:
            k_outputs = None
        return outputs, k_outputs  # a, present, (attentions)


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):
        output_attn = self.attn(self.ln_1(x),
                                layer_past=layer_past,
                                attention_mask=attention_mask,
                                head_mask=head_mask)
        a = output_attn[0]  # output_attn: a, present, (attentions)

        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m

        outputs = [x] + output_attn[1:]
        return outputs  # x, present, (attentions)


class KG_Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False, add_or_concat='add'):
        super(KG_Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = KG_Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

        self.memory_layer_norm = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.ff_layer = MLP(4 * nx, config)
        self.kmlp = nn.Linear(2 * nx, nx, bias=True)
        self.add_or_concat = add_or_concat

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, pred_kg=None,
                pred_kg_attention_mask=None, copy=False, ):
        output_attn, k_output_attn = self.attn(self.ln_1(x),  # torch.Size([5, 110, 768])
                                               layer_past=layer_past,
                                               attention_mask=attention_mask,
                                               head_mask=head_mask,
                                               pred_kg=self.ln_1(pred_kg) if pred_kg is not None else None,
                                               pred_kg_attention_mask=pred_kg_attention_mask if pred_kg_attention_mask is not None else None,
                                               copy=copy, )
        a = output_attn[0]
        # output_attn: 1. a - context vector torch.Size([5, 110, 768]) , 2. present, 3. attentions
        if k_output_attn is not None:
            k = k_output_attn[0]
        else:
            k = 0

        if pred_kg is not None:
            if self.add_or_concat == 'add':
                # todo: 使用 memory vector 更新 output vector
                res = self.memory_layer_norm(a + k)
                # res = self.ff_layer(res)
            else:
                # concat a from text sequence and k from knowledge sequence
                res = torch.cat((a, k), dim=2)  # todo: 拼接还是相加？？？？
                res = self.kmlp(res)  # todo: add layer norm
        else:
            res = a

        # x = x + a
        x = x + res
        m = self.mlp(self.ln_2(x))
        x = x + m  # hidden state

        # outputs = [x] + output_attn[1:]
        # return outputs  # x, present, attention_weights
        if k_output_attn is not None:
            return [x, output_attn[1:], k_output_attn[1:]]
        else:
            return [x, output_attn[1:], None]
        # outputs, (present, attention weights)


class GPT2PreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = GPT2Config
    pretrained_model_archive_map = GPT2_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super(GPT2PreTrainedModel, self).__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def reorder_encoder_out(self, encoder_outs, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_encoder_outs = []
        for encoder_out in encoder_outs:
            new_encoder_outs.append(encoder_out.index_select(0, new_order))
        return new_encoder_outs


GPT2_START_DOCSTRING = r"""    OpenAI GPT-2 model was proposed in
    `Language Models are Unsupervised Multitask Learners`_
    by Alec Radford*, Jeffrey Wu*, Rewon Child, David Luan, Dario Amodei** and Ilya Sutskever**.
    It's a causal (unidirectional) transformer pre-trained using  language modeling on a very large
    corpus of ~40 GB of text data.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`Language Models are Unsupervised Multitask Learners`:
        https://openai.com/blog/better-language-models/

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.GPT2Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

GPT2_INPUTS_DOCSTRING = r"""    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            GPT-2 is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.
            Indices can be obtained using :class:`transformers.GPT2Tokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **past**:
            list of ``torch.FloatTensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `past` output below). Can be used to speed up sequential decoding.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            A parallel sequence of tokens (can be used to indicate various portions of the inputs).
            The embeddings from these tokens will be summed with the respective token embeddings.
            Indices are selected in the vocabulary (unlike BERT which has a specific vocabulary for segment indices).
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""


@add_start_docstrings("The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
                      GPT2_START_DOCSTRING, GPT2_INPUTS_DOCSTRING)
class GPT2Model(GPT2PreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the last layer of the model.
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """

    def __init__(self, config, source_length=0):
        super(GPT2Model, self).__init__(config)
        print("source length: {}".format(source_length))
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        # self.output_past = config.output_past

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.source_length = source_length

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        self.wte = self._get_resized_embeddings(self.wte, new_num_tokens)
        return self.wte

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1])
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            # attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            attention_mask = attention_mask.to(dtype=torch.float32)
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        # source mask:
        if self.source_length > 0:
            head_mask = torch.ones(1, 1, 1, input_ids.size(1), dtype=torch.float, device=input_ids.device)
            head_mask[..., :self.source_length] = 0
            head_mask = [head_mask] * self.config.n_layer

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0

        hidden_states = inputs_embeds + position_embeds + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(hidden_states,
                            layer_past=layer_past,
                            attention_mask=attention_mask,
                            head_mask=head_mask[i])

            hidden_states, present = outputs[:2]

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        # if self.output_past:
        #    outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, (presents), (all hidden_states), (attentions)


@add_start_docstrings("""The GPT2 Model transformer with a language modeling head on top
(linear layer with weights tied to the input embeddings). """, GPT2_START_DOCSTRING, GPT2_INPUTS_DOCSTRING)
class GPT2LMHeadModel(GPT2PreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-1`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import torch
        from transformers import GPT2Tokenizer, GPT2LMHeadModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

    """

    def __init__(self, config, source_length=0):
        super(GPT2LMHeadModel, self).__init__(config)
        self.transformer = GPT2Model(config, source_length)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.transformer.wte)

    def get_representation(self, input_ids, position_ids=None, attention_mask=None):
        '''get sentence representation via max-pooling'''
        transformer_outputs = self.transformer(input_ids,
                                               past=None,
                                               attention_mask=attention_mask,
                                               token_type_ids=None,
                                               position_ids=position_ids,
                                               head_mask=None)

        hidden_states = transformer_outputs[0]
        pool_repr = torch.max(hidden_states * attention_mask.unsqueeze(2).expand_as(hidden_states.data), 1)[0]
        return pool_repr

    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None):
        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask)
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits  # [..., :-1, :].contiguous()
            shift_labels = labels  # [..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)


@add_start_docstrings("""The GPT2 Model transformer with a language modeling and a multiple-choice classification
head on top e.g. for RocStories/SWAG tasks. The two heads are two linear layers.
The language modeling head has its weights tied to the input embeddings,
the classification head takes as input the input of a specified classification token index in the input sequence).
""", GPT2_START_DOCSTRING, GPT2_INPUTS_DOCSTRING)
class GPT2DoubleHeadsModel(GPT2PreTrainedModel):
    r"""
        **mc_token_ids**: (`optional`, default to index of the last token of the input) ``torch.LongTensor`` of shape ``(batch_size, num_choices)``:
            Index of the classification token in each input sequence.
            Selected in the range ``[0, input_ids.size(-1) - 1[``.
        **lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-1`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        **mc_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **lm_loss**: (`optional`, returned when ``lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Language modeling loss.
        **mc_loss**: (`optional`, returned when ``multiple_choice_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Multiple choice classification loss.
        **lm_prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **mc_prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)``
            Prediction scores of the multiplechoice classification head (scores for each choice before SoftMax).
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import torch
        from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

        # Add a [CLS] to the vocabulary (we should train it also!)
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
        print(tokenizer.cls_token_id, len(tokenizer))  # The newly token the last token of the vocabulary

        choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        encoded_choices = [tokenizer.encode(s) for s in choices]
        cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

        input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
        mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

        outputs = model(input_ids, mc_token_ids=mc_token_ids)
        lm_prediction_scores, mc_prediction_scores = outputs[:2]

    """

    def __init__(self, config):
        super(GPT2DoubleHeadsModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.transformer.wte)

    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                mc_token_ids=None, lm_labels=None, mc_labels=None):
        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask)

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        outputs = (lm_logits, mc_logits) + transformer_outputs[1:]
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)),
                            mc_labels.view(-1))
            outputs = (loss,) + outputs
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (lm loss), (mc loss), lm logits, mc logits, presents, (all hidden_states), (attentions)


class KG_GPT2Model(GPT2PreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the last layer of the model.
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """

    def __init__(self, config, source_length=0):
        super(KG_GPT2Model, self).__init__(config)
        print("source length: {}".format(source_length))
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # self.h = nn.ModuleList([KG_Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])

        self.blocks = []
        for idx in range(config.n_layer):
            if idx != config.n_layer - 1:
                self.blocks.append(Block(config.n_ctx, config, scale=True))
            else:
                self.blocks.append(KG_Block(config.n_ctx, config, scale=True))
        self.h = nn.ModuleList(self.blocks)

        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.source_length = source_length

        self.init_weights()
        self.num_heads = config.n_head

    def _resize_token_embeddings(self, new_num_tokens):
        self.wte = self._get_resized_embeddings(self.wte, new_num_tokens)
        return self.wte

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, pred_kg=None, pred_kg_rel_ids=None, pred_kg_attention_mask=None,
                pred_kg_position_ids=None, copy=False, embed_or_hidden='embed'):

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1])  # # [batch_size, to_seq_length]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, to_seq_length]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            attention_mask = attention_mask.to(dtype=torch.float32)
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        # source mask:
        if self.source_length > 0:
            '''
            self-attention使用，可选，想用哪些head，就为1或者None，不想用的head就为0。shape为[num_heads] or [num_hidden_layers x num_heads]，即：可以每层每个head单独设置mask。
            '''
            head_mask = torch.ones(1, 1, 1, input_ids.size(1), dtype=torch.float, device=input_ids.device)
            head_mask[..., :self.source_length] = 0  # gpt2是decoder，所以source部分的head不需要用
            head_mask = [head_mask] * self.config.n_layer

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0

        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)  # （bsz, inp_len, dim）

        # todo: for knowledge memory
        if pred_kg is not None:  # (bsz, kg_len)
            kg_memo_shape = pred_kg.size()
            pred_kg = pred_kg.view(-1, kg_memo_shape[-1])
            if pred_kg_rel_ids is not None:
                pred_kg_rel_ids = pred_kg_rel_ids.view(-1, kg_memo_shape[-1])
            if pred_kg_position_ids is not None:
                pred_kg_position_ids = pred_kg_position_ids.view(-1, kg_memo_shape[-1])

            if pred_kg_position_ids is None:
                pred_kg_position_ids = torch.arange(past_length, pred_kg.size(-1) + past_length, dtype=torch.long,
                                                    device=pred_kg.device)
                pred_kg_position_ids = pred_kg_position_ids.unsqueeze(0).expand_as(pred_kg)

            if pred_kg_attention_mask is not None:  # (bsz, kg_len)
                pred_kg_attention_mask = pred_kg_attention_mask.unsqueeze(1).unsqueeze(2)
                pred_kg_attention_mask = pred_kg_attention_mask.to(dtype=torch.float32)
                pred_kg_attention_mask = (1.0 - pred_kg_attention_mask) * -10000.0

            # 使用 text generator 的 embedding layer 进行编码
            pred_kg_embeds = self.wte(pred_kg)
            pred_kg_position_embeds = self.wpe(pred_kg_position_ids)
            if pred_kg_rel_ids is not None:
                pred_kg_rel_embeds = self.wte(pred_kg_rel_ids)
            else:
                pred_kg_rel_embeds = 0

            pred_kg_hidden = pred_kg_embeds + pred_kg_position_embeds + pred_kg_rel_embeds
            pred_kg_hidden = self.drop(pred_kg_hidden)
            kg_memo_shape = kg_memo_shape + (pred_kg_hidden.size(-1),)  # （bsz, inp_len, dim）
        else:
            pred_kg_hidden = None
            pred_kg_attention_mask = None

        presents = ()
        all_attentions = []
        all_kg_attentions = []
        all_hidden_states = ()

        kg_outputs = [0]
        for i, (block, layer_past) in enumerate(zip(self.h, past)):  # 12 layers
            if self.output_hidden_states:  # false
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            # todo: 从kg memory 获得的 vector 是与hidden_states 相加 还是 拼接呢？
            if i != len(self.h) - 1:  # 前 11 layers
                outputs = block(hidden_states,
                                layer_past=layer_past,
                                attention_mask=attention_mask,
                                head_mask=head_mask[i])
                if embed_or_hidden == 'hidden':
                    kg_outputs = block(pred_kg_hidden,
                                       layer_past=layer_past,
                                       attention_mask=pred_kg_attention_mask,
                                       head_mask=head_mask[i][:, :, :, :pred_kg_hidden.size(1)],)

            else:  # layer 12
                outputs = block(hidden_states,
                                layer_past=layer_past,  # None
                                attention_mask=attention_mask,
                                head_mask=head_mask[i],
                                pred_kg=kg_outputs[0] if embed_or_hidden=='hidden' else pred_kg_hidden,
                                pred_kg_attention_mask=pred_kg_attention_mask,
                                copy=copy,)

            # hidden_states, present = outputs[:2]
            hidden_states = outputs[0]

            if copy and pred_kg_hidden is not None and i == len(self.h) - 1:
                kg_attention = outputs[-1][1]
                kg_attention = kg_attention.view(input_shape[0], self.num_heads, hidden_states.size(1),
                                                 pred_kg_hidden.size(1))
                kg_attention = kg_attention[:, 0, :, :]
                all_kg_attentions.append(kg_attention)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        # if self.output_past:
        #    outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        if copy and pred_kg_hidden is not None:
            outputs = outputs + (all_kg_attentions,)

        return outputs  # last hidden state, (presents), (all hidden_states), (attentions)


class TextGen(GPT2PreTrainedModel):
    def __init__(self, config, source_length=0, tokenizer=None):
        super(TextGen, self).__init__(config)
        self.transformer = KG_GPT2Model(config, source_length)
        self.tokenizer = tokenizer

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()
        self.tie_weights()
        self.diverter = nn.Linear(config.n_embd, 2)

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        logger.info("Tie weights in head!!!!!")
        self._tie_or_clone_weights(self.lm_head,
                                   self.transformer.wte)

    def forward(self, src_input_ids, src_input_type_ids, attention_mask, src_position_ids,
                target_input_ids, target_input_type_ids, target_position_ids, labels,
                pred_kg=None, pred_kg_rel_ids=None, pred_kg_attention_mask=None, pred_kg_position_ids=None,
                copy=False, embed_or_hidden='embed'):

        input_ids = torch.cat([src_input_ids, target_input_ids], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(target_input_ids).to(target_input_ids.device)],
                                   dim=1)
        position_ids = torch.cat([src_position_ids, target_position_ids], dim=1)
        type_ids = torch.cat([src_input_type_ids, target_input_type_ids], dim=1)

        lm_probs = self.autoreg_forward(input_ids, type_ids, attention_mask, position_ids,
                                        pred_kg=pred_kg,
                                        pred_kg_rel_ids=pred_kg_rel_ids,
                                        pred_kg_attention_mask=pred_kg_attention_mask,
                                        pred_kg_position_ids=pred_kg_position_ids,
                                        copy=copy,
                                        embed_or_hidden=embed_or_hidden,)
        '''
        Compute loss: generation loss
        '''
        gen_loss_fn = nn.NLLLoss(ignore_index=-1, reduction='mean')  # 不计算 source部分 和 pad 部分
        lm_probs_clamp = lm_probs.clamp(min=1e-5)
        gen_loss = gen_loss_fn(lm_probs_clamp.log().view(-1, lm_probs.size(-1)), labels.view(-1))
        assert (not torch.isinf(gen_loss).any().item() and not torch.isnan(gen_loss).any().item())

        return gen_loss

    def autoreg_forward(self, input_ids, type_ids, attention_mask, position_ids, do_generate=False,
                        pred_kg=None, pred_kg_rel_ids=None, pred_kg_attention_mask=None, pred_kg_position_ids=None,
                        is_infer=False, temperature=1.0, copy=False, embed_or_hidden='embed'):
        '''
        return:
            - probs: bsz x L x vocab
        '''
        outputs = self.transformer(input_ids,
                                   token_type_ids=type_ids,
                                   attention_mask=attention_mask,
                                   position_ids=position_ids,
                                   pred_kg=pred_kg,
                                   pred_kg_rel_ids=pred_kg_rel_ids,
                                   pred_kg_attention_mask=pred_kg_attention_mask,
                                   pred_kg_position_ids=pred_kg_position_ids,
                                   copy=copy,
                                   embed_or_hidden=embed_or_hidden,)
        hidden_states = outputs[0]  # the last hidden state

        if do_generate:  # 测试的时候
            hidden_states = hidden_states[:, -1, :].unsqueeze(1)  # the last hidden state of the last word

        softmax = nn.Softmax(dim=-1)

        if copy and pred_kg is not None:
            kg_attention_weights = outputs[1][-1]

            # todo: 计算 复制门 \gama_t
            gates = F.softmax(self.diverter(hidden_states), -1)
            gen_gate, copy_gate = gates.chunk(2, dim=-1)

            lm_probs = gen_gate * softmax(self.lm_head(hidden_states) / temperature)

            bsz, tgt_len, _ = hidden_states.size()
            index = pred_kg.contiguous().view(1, bsz, -1).expand(tgt_len, -1, -1)  # (tgt_len, bsz, mem_len)
            copy_probs = (copy_gate * kg_attention_weights[-1]).view(tgt_len, bsz, -1)  # (tgt_len, bsz, mem_len)

            lm_probs = lm_probs.transpose(0, 1).scatter_add_(-1, index, copy_probs)
            lm_probs = lm_probs.transpose(0, 1)

            if torch.sum(torch.isnan(lm_probs)) != 0:
                pdb.set_trace()
        else:
            lm_logits = self.lm_head(hidden_states)
            lm_probs = softmax(lm_logits / temperature)

        return lm_probs


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, sum=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if sum:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class KnowledgeTextGen(GPT2PreTrainedModel):
    def __init__(self, config, source_length=0, tokenizer=None):
        super(KnowledgeTextGen, self).__init__(config)
        self.transformer_tg = TextGen(config, source_length=source_length, tokenizer=tokenizer)
        self.tokenizer = tokenizer

    # for training
    def forward(self, src_input_ids, src_input_type_ids, attention_mask, src_position_ids,
                target_input_ids, target_input_type_ids, target_position_ids, labels,
                pred_kg=None, pred_kg_rel_ids=None, pred_kg_attention_mask=None, pred_kg_position_ids=None,
                copy=False, embed_or_hidden='embed',temperature=1.0,):

        tg_loss = self.transformer_tg(src_input_ids, src_input_type_ids, attention_mask, src_position_ids,
                                      target_input_ids, target_input_type_ids, target_position_ids, labels,
                                      pred_kg, pred_kg_rel_ids, pred_kg_attention_mask, pred_kg_position_ids,
                                      copy=copy, embed_or_hidden=embed_or_hidden, temperature=temperature,)  # (bsz, l, vocab)

        assert torch.sum(torch.isnan(tg_loss)).item() == 0, pdb.set_trace()
        return tg_loss

    # for inference
    def autoreg_generate(self, src_input_ids, src_input_type_ids, attention_mask, src_position_ids, seq_generator,
                         pred_kg=None, pred_kg_rel_ids=None, pred_kg_attention_mask=None, pred_kg_position_ids=None,
                         tgt_bos_rel_ids=None, is_end_at_relation=None, greedy_or_beam="greedy", copy=False, ):
        kg_hypos, multi_kg_hypos = {"prediction": []}, None

        tg_sample = {"input_ids": src_input_ids,
                     "input_type": src_input_type_ids,
                     "attention_mask": attention_mask,
                     "position_ids": src_position_ids,
                     "pred_kg": pred_kg,
                     "pred_kg_rel_ids": pred_kg_rel_ids,
                     "pred_kg_attention_mask": pred_kg_attention_mask,
                     "pred_kg_position_ids": pred_kg_position_ids,}

        if greedy_or_beam == "greedy":
            tg_hypos, multi_tg_hypos = seq_generator.greedy_generate(self.transformer_tg.autoreg_forward, tg_sample)
        else:
            tg_hypos, multi_tg_hypos = seq_generator.beam_generate(self.transformer_tg.autoreg_forward, tg_sample)

        return kg_hypos['prediction'], tg_hypos, multi_kg_hypos, multi_tg_hypos

