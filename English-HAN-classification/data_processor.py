#import jieba
import re
import numpy as np
#from pyltp import SentenceSplitter
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
from torchtext import data
from torchtext.vocab import Vectors

import torch
import pandas as pd

#nltk.download('stopwords')
#nltk.download('punkt')

from nltk.corpus import stopwords
list_stopWords = list(set(stopwords.word('english')))

#def tokenizer(text): # create a tokenizer function
#    regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
#    text = regex.sub(' ', text)
#    return [word for word in jieba.cut(text) if word.strip()]

# 去停用词
#def get_stop_words():
#    file_object = open('./ose_data/english.txt')
#    stop_words = []
#    for line in file_object.readlines():
#        line = line[:-1]
#        line = line.strip()
#        stop_words.append(line)
#    return stop_words

def load_data(args):
    print('加载数据中...')
    #stop_words = get_stop_words() # 加载停用词表
    '''
    如果需要设置文本的长度，则设置fix_length,否则torchtext自动将文本长度处理为最大样本长度
    text = data.Field(sequential=True, tokenize=tokenizer, fix_length=args.max_len, stop_words=stop_words)
    '''
    text = data.Field(sequential=True, lower=True, tokenize=word_tokenizer, stop_words=list_stopWords)
    #text = data.Field(tokenize='spacy', lower=True)
    label = data.Field(sequential=False)

    #text.tokenize = tokenizer
    train, val = data.TabularDataset.splits(
            path='ose_data/',
            skip_header=True,
            train='train.tsv',
            validation='validation.tsv',
            format='tsv',
            fields=[('index', None), ('label', label), ('text', text)],
        )

    if args.static:
        text.build_vocab(train, val, vectors=Vectors(name='/home/yuling/tyling-data/word_vectors/glove.6B.100d.txt')) # 此处改为你自己的词向量
        args.embedding_dim = text.vocab.vectors.size()[-1]
        args.vectors = text.vocab.vectors
    else: text.build_vocab(train, val)

    label.build_vocab(train, val)
    args.vocab_size = len(text.vocab)
    args.label_num = len(label.vocab)

    return text

#
'''
产生batch_size大小的文章,并且将每篇文章的句子数量和每个句子的单词数量pad成相同的长度
不同batch之间的max_words, max_sents可以不一样,但相同batch中max_words, max_sents必须一致
'''
def gen_batch(args, TEXT):
    batch_size = args.batch_size
    df = pd.read_csv('ose_data/train.tsv', sep='\t')
    inputs = df.values
    data_length = len(inputs)
    args.iterations = data_length//batch_size
    while True:
        r_num = np.random.randint(data_length-batch_size+1)
        batch_docs = inputs[r_num:r_num+batch_size]
        batch_docs, batch_targets = pad_batch(batch_docs, TEXT)
        yield batch_docs, batch_targets

def pad_batch(batch_docs, TEXT):
    res_batch_docs = []
    max_words, max_sents = 0, 0
    res_batch_targets = []
    for doc in batch_docs:
        doc_text = doc[2]
        res_doc_text = []
        # 使用LTP将一篇文章划分成若干句子
        #sents = SentenceSplitter.split(doc_text)
        sents=sent_tokenize(doc_text)
        max_sents = max(max_sents, len(sents))
        for i, sent in enumerate(sents):
            sent = TEXT.preprocess(sent)
            sent = [TEXT.vocab.stoi[word] for word in sent]
            max_words = max(max_words, len(sent))
            res_doc_text.append(sent)
        res_batch_docs.append(res_doc_text)
        res_batch_targets.append(doc[1])

    for doc in res_batch_docs:
        sents = doc
        for sent in sents:
            while len(sent) < max_words: sent.append(0)
        while len(sents) < max_sents:
            sents.append([0 for _ in range(max_words)])
    return torch.LongTensor(res_batch_docs), torch.LongTensor(res_batch_targets)

def gen_test(TEXT, args):
    batch_size = args.batch_size
    df = pd.read_csv('ose_data/validation.tsv', sep='\t')
    inputs = df.values
    data_length = len(inputs)
    while True:
        r_num = np.random.randint(data_length-batch_size+1)
        batch_docs = inputs[r_num:r_num+batch_size]
        batch_docs, batch_targets = pad_batch(batch_docs, TEXT)
        yield batch_docs, batch_targets



