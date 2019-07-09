#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
nmt.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""

"""
这个模块有三个功能，
pad_sents：统一句子长度。
read_corpus：读取语料库，把一句话存到一个list中去。
batch_iter：以(key,value)保存数据，决定我们一次算多少个数据。不了解的去看下SGD，BGD就好，这里用的是BGD。
"""
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



# 这个函数的目的是统一所有句子长度，先找出最长的句子长度，之后在长度不够的句子末尾补pad_token。
def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    ### YOUR CODE HERE (~6 Lines)
    max_len = max(len(sent) for sent in sents)
    for sent in sents:
        sents_padded.append(sent + [pad_token] * max(0, max_len - len(sent)))
    ### END YOUR CODE

    return sents_padded


# 读取语料库，按行读取语料库，每一句话是一个list，把每一个词存入list[i]。
def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data

# 一个生成器，会不断生成数据，batch_size是每一次运行数据数量。data是一个[(x,y),...,(x,y)]。
def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents
