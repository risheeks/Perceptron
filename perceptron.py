#!/usr/bin/env python
# coding: utf-8
"""© 2019 Rajkumar Pujari All Rights Reserved
    
    - Original Version
    
    Author: Rajkumar Pujari
    Last Modified: 03/12/2019
    
    """
import numpy as np
from classifier import BinaryClassifier
from random import choice
from utils import get_feature_vectors
from random import shuffle
import random

class Perceptron(BinaryClassifier):
    
    def __init__(self, args):
        #TO DO: Initialize parameters here
        self.lr = args.lr
        self.f_dim = args.f_dim
        self.num_iter = args.num_iter
        self.vocab_size = args.vocab_size
        self.bin_feats = args.bin_feats
        self.weight = [0] * self.f_dim
        self.bias = 0
    #        print(self.weight)
    #        print(args.lr)
    #        raise NotImplementedError
    
    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
#        print(train_data[0])
        tr_size = len(train_data[0])
        indices = list(range(tr_size))
        random.seed(1) #this line is to ensure that you get the same shuffled order everytime
        random.shuffle(indices)
        train_data = ([train_data[0][i] for i in indices], [train_data[1][i] for i in indices])
        x_vectors = get_feature_vectors(train_data[0], self.bin_feats)
#        print(train_data[1])
        for i in range(self.num_iter):
            """predicts = self.predict(train_data[0])
            
            j = 0
            for pre in predicts:
                expected_value = train_data[1][j]
                j += 1
                if pre != expected_value:
                    self.weight = self.weight + np.dot((self.lr * expected_value), x_vectors[j])
                    self.bias = self.bias + self.lr * expected_value
                    print('false')
                else:
                    print('true')
            """
            
            #            print(choice(train_data[1]))
            
#            x = choice(train_data[0])
            for x in train_data[0]:
#                x = choice(train_data[0])
                expected_value = train_data[1][train_data[0].index(x)]
                x_vector = x_vectors[train_data[0].index(x)]
                product = np.dot(self.weight, x_vector)
                total = product + self.bias
#                print(train_data[0].index(x))
                if total > 0:
                    total = 1
                else:
                    total = -1

#                print('total=' + str(total) + ' expected_val= ' + str(expected_value))
                if total != expected_value:
                    self.weight = self.weight + np.dot((self.lr * expected_value), x_vector)
                    self.bias = self.bias + self.lr * expected_value
#                    print('false')
#                else:
#                    print('true')


#            print(x)
#            print(x_vector)
#            print(expected_value)

#        print(train_data[1])
#        raise NotImplementedError

    def predict(self, test_x):
        #TO DO: Compute and return the output for the given test inputs
        x_vectors = get_feature_vectors(test_x, self.bin_feats)
        ret_vector = []
        for vector in x_vectors:
            product = np.dot(self.weight, vector)
            total = product + self.bias
#            print(total)
            if total > 0:
                total = 1
            else:
                total = -1
            ret_vector.append(total)
#        print(ret_vector)
        return ret_vector

    def evaluate(self, test_data):
        test_x, test_y = test_data
        pred_y = self.predict(test_x)
        tp, tn, fp, fn = 0., 0., 0., 0.
        for py, gy in zip(pred_y, test_y):
            if py == -1 and gy == -1:
                tn += 1
            elif py == -1 and gy == 1:
                fn += 1
            elif py == 1 and gy == 1:
                tp += 1
            elif py == 1 and gy == -1:
                fp += 1
        cm = ((tn, fn), (fp, tp))
        return self.metrics(cm)

    def metrics(self, confusion_matrix):
        true_positives = confusion_matrix[1][1]
        false_positives = confusion_matrix[1][0]
        false_negatives = confusion_matrix[0][1]
        true_negatives = confusion_matrix[0][0]
        total_size = true_positives + true_negatives + false_positives + false_negatives
        acc = 100 * (true_positives + true_negatives) / (total_size * 1.0)
        prec = 100 *  (true_positives * 1.0) / (true_positives + false_positives + 0.01)
        rec = 100 * (true_positives * 1.0) / (true_positives + false_negatives + 0.01)
        if prec == 0 and rec == 0:
            f1 = 0
        else:
            f1 = (2.0 * prec * rec) / (prec + rec)
        return (acc, prec, rec, f1)


class AveragedPerceptron(BinaryClassifier):
    
    def __init__(self, args):
        self.lr = args.lr
        self.f_dim = args.f_dim
        self.num_iter = args.num_iter
        self.vocab_size = args.vocab_size
        self.bin_feats = args.bin_feats
        self.weight = [0] * self.f_dim
        self.bias = 0
        self.survival = 0
    
    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
        tr_size = len(train_data[0])
        indices = list(range(tr_size))
        random.seed(1) #this line is to ensure that you get the same shuffled order everytime
        random.shuffle(indices)
        train_data = ([train_data[0][i] for i in indices], [train_data[1][i] for i in indices])
        x_vectors = get_feature_vectors(train_data[0], False)
        for i in range(self.num_iter):
            for y in train_data[0]:
                x = choice(train_data[0])
                expected_value = train_data[1][train_data[0].index(x)]
                x_vector = x_vectors[train_data[0].index(x)]
                total = np.dot(self.weight, x_vector) + self.bias
#                print(train_data[0].index(x))
                if total > 0:
                    total = 1
                else:
                    total = -1
                if total != expected_value:
                    weight1 = self.weight + np.dot((self.lr * expected_value), x_vector)
                    self.weight = (np.array(np.dot(self.survival, self.weight)) + np.array(weight1)) / (self.survival + 1)
                    self.bias = self.bias + self.lr * expected_value
#                    print(self.survival)
#                    print('false')
                    self.survival = 1
                else:
                    self.survival += 1
#                    print('true')

    
    def predict(self, test_x):
        #TO DO: Compute and return the output for the given test inputs
        x_vectors = get_feature_vectors(test_x, False)
        ret_vector = []
        for vector in x_vectors:
            product = np.dot(self.weight, vector)
            total = product + self.bias
            #            print(total)
            if total > 0:
                total = 1
            else:
                total = -1
            ret_vector.append(total)
    #        print(ret_vector)
        return ret_vector

