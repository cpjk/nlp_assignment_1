# -*- coding: utf-8 -*-
import pdb

from nltk.util import ngrams
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import numpy as np
import random as rand

START_TAG = "<s>"
END_TAG = "</s>"


# with open('t8.shakespeare.txt', 'r') as f:
#     corpus = f.read().split(" ")

# corpus = filter(lambda w: w != '', corpus)

# unigrams = list(ngrams(corpus, 1))
# bigrams = list(ngrams(corpus, 2))

# unigram_cntr = Counter(unigrams)
# bigram_cntr = Counter(bigrams)

# common_uni = unigram_cntr.most_common(15)
# common_bi = bigram_cntr.most_common(15)

# print "Unigrams:"
# for unigram, count in common_uni:
#     print unigram

# print "Bigrams:"
# for bigram, count in common_bi:
#     print bigram


# part 2

with open('t8.shakespeare.txt', 'r') as f:
    corpus = f.read()

corpus = filter(lambda w: w != '', corpus)

# surround sentences with start and end tags
sents = sent_tokenize(corpus)
sents = map(lambda s: word_tokenize(s), sents)
sents = map(lambda s: [START_TAG] + s + [END_TAG], sents)

# flatten sentence word-lists
corpus = [token for sent in sents for token in sent]

# generate unigrams and bigrams
unigrams = list(ngrams(corpus, 1))
bigrams = list(ngrams(corpus, 2))

# get list of first words in each bigram. for each of these, store the words following it
# for each first word, store counts of each word that follows it
bigram_words = {}
for (first_w, sec_w) in bigrams:
    bigram_words[first_w] = bigram_words.get(first_w, {})
    bigram_words[first_w][sec_w] = bigram_words[first_w].get(sec_w, 0) + 1

# for each first word, calc cond probs of every word that follows it
word_probs = {}
for first_w in bigram_words.keys():
    word_probs[first_w] = {}
    total_sec_words = reduce(lambda sm, elem: sm + elem, bigram_words[first_w].values())

    for sec_w in bigram_words[first_w].keys():
        sec_w_prob = float(bigram_words[first_w][sec_w]) / total_sec_words
        word_probs[first_w][sec_w] = sec_w_prob

# for each first word, build a range of probabilities of second words
prob_lists = {}
for first_w in word_probs.keys():
    prob_lists[first_w] = {}
    curr_place = 0
    for sec_w in word_probs[first_w].keys():
        prob_range = [curr_place, curr_place + word_probs[first_w][sec_w]] # (curr_place, sec_w prob]
        prob_lists[first_w][sec_w] = prob_range
        curr_place = curr_place + word_probs[first_w][sec_w]


# prev_word = '<s>'
# print prev_word
# while prev_word != '</s>':
#     rand_num = rand.random()
#     for sec_w in prob_lists[prev_word].keys():
#         low, hi = prob_lists[prev_word][sec_w]
#         if low <= rand_num <= hi: # select this word
#             prev_word = sec_w
#             print prev_word
#             break


def generate_sentence(prob_lists):
    sentence = []
    prev_word = '<s>'
    sentence.append(prev_word)

    while prev_word != '</s>':
        rand_num = rand.random()
        for sec_w in prob_lists[prev_word].keys():
            low, hi = prob_lists[prev_word][sec_w]
            if low <= rand_num <= hi: # select this word
                prev_word = sec_w
                sentence.append(prev_word)
                break
    return " ".join(sentence)
for x in list(range(10)):
    print generate_sentence(prob_lists)

pdb.set_trace()

# look at current word. gen random number. select random word that has followed that word in corpus

# start with start tag
# for 1..5 do
#   

