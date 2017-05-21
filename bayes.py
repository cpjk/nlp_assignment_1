#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pdb


class MyBayesClassifier():
    def __init__(self, smooth=1):
        self._smooth = smooth  # This is for additive smoothing
        self._feat_prob = []  # do not change the name of these vars
        self._class_prob = []
        self._num_classes = []
        self._num_features = []

    def train(self, X, y):
        """
        X is a document-term matrix.
        y is an array of classes where y[0] is class of X[0]
        """
        alpha_smooth = self._smooth

        classes, class_counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(classes, class_counts))

        num_classes, num_features = len(classes), X.shape[1]

        num_examples = X.shape[0]

        self._num_classes, self._num_features = num_classes, num_features
        self._feat_prob = np.zeros((num_classes, num_features))
        self._class_prob = np.zeros(num_classes)

        # get total num of examples. done

        # for each class, get number of examples having that class
        # class_counts = np.unique(y, return_counts=True)[1]

        # for each class, (get all examples in class)
        #   for each word in vocabulary
        #     get the number of examples that the word occurs in, for that class
        #     (sum all columns of the examples for the class

        pdb.set_trace()

        ex_for_class = {}
        for cls in classes:
            # array of examples of that class, each of len num_feat
            ex_for_class[cls] = np.zeros((class_counts[cls], num_features))

            np_idx = 0
            for idx, ex in enumerate(X):
                if y[idx] == cls:
                    ex_for_class[cls][np_idx] = ex
                    np_idx += 1
        pdb.set_trace()

        class_examples = np.zeros
        # for each class k, (get all examples in class)
        #   for each word wt in vocabulary
        #     estimate P(wt | C=k)

        # for each class k, get P(C=k)

        # - For each class y, for each feature x, find P(x=0|y). note: P(x=1|y) = 1 - P(x=0|y)
        # and store in self._feat_prob[class][feat]
        # then P(y|X[i]) = self._class_prob[y] * mult(self._feat_prob[y]


    # for each feature vector in X, the corresponding index in pred should be
    # 0 if positive, 1 if negative
    def predict(self, X):
        pred = np.zeros(len(X))
        # your code goes here
        return pred

    @property
    def probs(self):
        # please leave this intact, we will use it for marking
        return self._class_prob, self._feat_prob

    # def __calc_class_probs(self, classes, class_occurrences, num_examples):
    #     # - Find prob of each class P(y), store in self._class_prob[classes.index_of(y)]
    #     #   P(y) = num_occurrences_of_y / num_examples
    #     for cls, index in classes:
    #         self._class_prob[index] = class_occurrences[index] / num_examples



# our feature set is a vector of word counts for the vocabulary
# e.g. X_train[i][j] is count of word j in example i

"""
Here is the calling code

"""

with open('sentiment_data/rt-polarity_utf8.neg', 'r') as f:
    lines_neg = f.read().splitlines()

with open('sentiment_data/rt-polarity_utf8.pos', 'r') as f:
    lines_pos = f.read().splitlines()

data_train = lines_neg[0:5000] + lines_pos[0:5000] # first 5000 neg, second 5000 pos
data_test = lines_neg[5000:] + lines_pos[5000:] # first 331 neg, second 331 pos

y_train = np.append(np.ones((1, 5000)), (np.zeros((1, 5000))))
y_test = np.append(np.ones((1, 331)), np.zeros((1, 331)))

# You will be changing the parameters to the CountVectorizer below

# max_df 1.0 ignores a word when more than 100% of documents contain
# it (impossible)
# min_df 1 ignores a word when less than 1 document contains it (impossible)
vectorizer = CountVectorizer(lowercase=True, stop_words=None,  max_df=1.0,
                             min_df=1, max_features=None,  binary=True)

# learn the vocabulary of the training data (fit), and return sparse
# term count matrix
X_train = vectorizer.fit_transform(data_train).toarray()

X_test = vectorizer.transform(data_test).toarray()

feature_names = vectorizer.get_feature_names()

clf = MyBayesClassifier(1)
clf.train(X_train, y_train)
y_pred = clf.predict(X_test)
print np.mean((y_test - y_pred) == 0) # assert no diff between y_test and y_pred
