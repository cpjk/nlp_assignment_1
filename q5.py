# -*- coding: utf-8 -*-
import pdb

from nltk.util import ngrams
from collections import Counter


with open('t8.shakespeare.txt', 'r') as f:
    corpus = f.read().split(" ")

corpus = filter(lambda w: w != '', corpus)

unigrams = list(ngrams(corpus, 1))
bigrams = list(ngrams(corpus, 2))

unigram_cntr = Counter(unigrams)
bigram_cntr = Counter(bigrams)

common_uni = unigram_cntr.most_common(15)
common_bi = bigram_cntr.most_common(15)

print "Unigrams:"
for unigram, count in common_uni:
    print unigram

print "Bigrams:"
for bigram, count in common_bi:
    print bigram
