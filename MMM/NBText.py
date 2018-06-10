# MLE for Multinomial Mixture model
# supervised and semi-supervised
#
# @nghia n h | Yamada lab

import os
import sys
import logging
from decimal import *

import numpy as np
from scipy import special
from namedlist import namedlist
from sklearn import metrics
from sklearn import model_selection

import exceptionHandle as SelfException

logger = logging.getLogger('NBText')
SelfException.LogHandler('NBText')


class Dataset(object):
    __doc__ = 'Basic data frame' \
              'this class only supports equal length feature vector input'

    def __init__(self, *args):
        """
        Init Dataset empty instance or from other Dataset (copy data)
        :param args:
        """
        try:
            if len(args) == 0 or args[0] is None:
                # data files
                self.map_file = ''
                self.train_file = ''
                self.test_file = ''

                # data structure
                self.train_x = np.empty((0))
                self.train_y = np.empty((0))
                self.test_x = np.empty((0))
                self.test_y = np.empty((0))
                self.class_name_list = []

                self.class_number = 0
                self.feature_number = 0
                self.train_number = 0
                self.test_number = 0
            elif len(args) == 1:
                # data files
                self.map_file = args[0].map_file
                self.train_file = args[0].train_file
                self.test_file = args[0].test_file

                # data structure
                self.train_x = args[0].train_x
                self.train_y = args[0].train_y
                self.test_x = args[0].test_x
                self.test_y = args[0].test_y
                self.class_name_list = args[0].class_name_list

                self.class_number = args[0].class_number
                self.feature_number = args[0].feature_number
                self.train_number = args[0].train_number
                self.test_number = args[0].test_number
            else:
                raise SelfException.DatasetInitArgsNumberViolated('More input arguments than required.')

        except SelfException.DatasetInitArgsNumberViolated as e:
            logger.exception('Dataset SelfException.DatasetInitArgsNumberViolated')
            e.recall_traceback(sys.exc_info())

        except BaseException:
            logger.exception('Dataset BaseException')
            raise

    def load_from_csv(self, file_name):
        """
        Load data from CSV file format
        :param file_name: list ordered file name [map,train,test]
        :return: none
        """
        self.map_file, self.train_file, self.test_file = file_name[:]

        loaded_train_data = np.genfromtxt(self.train_file, delimiter=',')
        self.train_x = np.mat(loaded_train_data.T[:-1].T)
        # reshape here is better than .T. Transpose does not change the data shape
        self.train_y = np.mat(np.reshape(loaded_train_data.T[-1], (-1,1)))

        loaded_test_data = np.genfromtxt(self.test_file, delimiter=',')
        self.test_x = np.mat(loaded_test_data.T[:-1].T)
        self.test_y = np.mat(np.reshape(loaded_test_data.T[-1], (-1,1)))

        loaded_map = np.genfromtxt(self.map_file, dtype='str', delimiter=',')
        self.class_name_list = loaded_map

        self.class_number = len(loaded_map)
        self.train_number, self.feature_number = np.shape(self.train_x)
        self.test_number = np.shape(self.test_x)[0]


class SslDataset(Dataset):
    __doc__ = 'Data frame for Semi-supervised Learning'

    def __init__(self, dataset=None, unlabeled_size=.5, random_seed=None):
        """
        This only take instance from exist Dataset and split labeled and unlabeled data by scaling size unlabeled_size
        The default scale is .5
        :param dataset: Base data, if dataset = None: set empty class.
        :param unlabeled_size: size of unlabeled data
        :param random_seed: numeric or None, seed of random splitting generator, default is None
        """
        super().__init__(dataset)
        # data structure
        self.train_xl = np.empty((0))
        self.train_yl = np.empty((0))
        self.train_xu = np.empty((0))
        self.train_yu = np.empty((0))

        self.train_labeled_number = 0
        self.train_unlabeled_number = 0

        self.unlabeled_size = unlabeled_size
        self.random_seed = random_seed

        try:
            if type(dataset) is Dataset:
                # split training data by unlabeled_size
                # TODO Becareful with StratifiedShuffleSplit
                sss = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=self.unlabeled_size,
                                                             random_state=self.random_seed)
                for labeled_indices, unlabeled_indices in sss.split(self.train_x, self.train_y):
                    self.train_xl = self.train_x[labeled_indices]
                    self.train_yl = self.train_y[labeled_indices, 0]
                    self.train_xu = self.train_x[unlabeled_indices]
                    self.train_yu = self.train_y[unlabeled_indices, 0]
                self.train_labeled_number = len(self.train_xl)
                self.train_unlabeled_number = len(self.train_xu)

            elif type(dataset) is SslDataset:
                # copy SslDataset
                self.train_xl = dataset.train_xl
                self.train_yl = dataset.train_yl
                self.train_xu = dataset.train_xu
                self.train_yu = dataset.train_yu
                self.train_labeled_number = dataset.train_labeled_number
                self.train_unlabeled_number = dataset.train_unlabeled_number
            elif dataset is not None:
                raise SelfException.DataTypeConstraint('SslDataset init: dataset datatype does not match')

        except SelfException.DataTypeConstraint as e:
            logger.exception('SslDataset SelfException.DataTypeConstraint')
            e.recall_traceback(sys.exc_info())

        except BaseException:
            logger.exception('SslDataset BaseException')
            raise


class Utility(object):
    __doc__ = 'Utilities'

    @staticmethod
    def log_factorial(x):
        """
        Compute ln of x factorial
        :return ln(x!):
        """
        return special.gammaln(np.array(x) + 1)

    @staticmethod
    def multinomial(word_vector, word_pr):
        """
        Compute multinomial density function
        :param word_vector: word vector
        :param word_pr: class conditional probability for word vector
        :return:
        """
        n = np.sum(word_vector)
        data, pr = np.array(word_vector), np.array(word_pr)
        result = Utility.log_factorial(n) - np.sum(Utility.log_factorial(data)) + np.sum(data * np.log(pr))
        # return np.exp(result)
        return Decimal(result).exp()

    @staticmethod
    def equal_sampling(component_number):
        """
        Return a list component_number elements of uniform samplings with constraint sum all element is 1
        :param component_number: number of component for of class
        :return: list of randomly sampling component for one class
        """
        samples = np.random.uniform(0, 1, component_number - 1)
        samples = np.append(samples, [0, 1])
        samples.sort()
        for i in range(len(samples) - 1):
            samples[i] = samples[i + 1] - samples[i]
        return samples[:-1]


class MultinomialNB(object):
    __doc__ = 'Deploy all labeled data Naive Bayes Multinomial model for text classification; ' \
              'MLE derivative solution; ' \
              'This only work with SslDataset data (using train_xl).'

    def __init__(self, dataset):
        try:
            if type(dataset) is not SslDataset: raise SelfException.DataTypeConstraint('Dataset type is not SslDataset')
            self.data = dataset

            # parameters set
            #  class prior probability [P(y1) ... P(yc)]
            self.prior_pr = np.zeros(self.data.class_number)
            #  word conditional probability per class [ [P(wi|y1)] ... [P(wi|yc)] ], i=1..d
            self.word_pr = np.zeros((self.data.class_number, self.data.feature_number))

            # predicted label
            # note: matrix type here, in the same type with data.test_y
            self.predicted_label = np.zeros((self.data.test_number, 1))

        except SelfException.DataTypeConstraint as e:
            logger.exception('MultinomialAllLabeled SelfException.DataTypeConstraint')
            e.recall_traceback(sys.exc_info())

        except BaseException:
            logger.exception('MultinomialAllLabeled BaseException')
            raise

    def train(self):
        """Training model"""
        for i in range(self.data.train_labeled_number):
            self.prior_pr[int(self.data.train_yl[i])] += 1
            self.word_pr[int(self.data.train_yl[i])] = self.word_pr[int(self.data.train_yl[i])] + self.data.train_xl[i]

        # add-one smoothing in use
        #  class prior probability
        self.prior_pr[:] += 1
        self.prior_pr = np.divide(self.prior_pr, float(self.data.train_labeled_number + self.data.class_number))

        #  word conditional probability
        sum_word_pr = self.word_pr.sum(axis=1)
        self.word_pr[:] += 1
        self.word_pr = (self.word_pr.T/(sum_word_pr + self.data.feature_number)).T

    def test(self):
        """Estimated value of x for each class"""
        estimate_value = [[Decimal(0) for j in range(self.data.class_number)] for i in range(self.data.test_number)]
        for i in range(self.data.test_number):
            max_val = 0
            max_id = -1
            for j in range(self.data.class_number):
                estimate_value[i][j] = Decimal(self.prior_pr[j]) * \
                                       Utility.multinomial(self.data.test_x[i,:], self.word_pr[j,:])
                if max_val < estimate_value[i][j]:
                    max_val = estimate_value[i][j]
                    max_id = j
            self.predicted_label[i, 0] = max_id


class MultinomialEM(object):
    __doc__ = 'Deploy Semi-supervised Multinomial model for text classification; ' \
              'EM algorithm; ' \
              'This only work with SslDataset data (using train_xl).'

    def __init__(self, dataset, theta_zero=None, epsilon=1e-4):
        """
        Init model
        :param dataset: dataset
        :param theta_zero: defalut None, set theta zero
        :param epsilon: default 1e-4, MLE convergence threshold
        """
        try:
            if type(dataset) is not SslDataset: raise SelfException.DataTypeConstraint('Dataset type is not SslDataset')
            self.data = dataset

            # parameters set
            #  class prior probability [P(y1) ... P(yc)]
            self.prior_pr = np.zeros(self.data.class_number)
            #  word conditional probability per class [ [P(wi|y1)] ... [P(wi|yc)] ], i=1..d
            self.word_pr = np.zeros((self.data.class_number, self.data.feature_number))
            #  init parameter set
            self.theta_zero = theta_zero
            self.EM_loop_count = -1
            self.epsilon = epsilon

            # predicted label
            # note: matrix type here, in the same type with data.test_y
            self.predicted_label = np.zeros((self.data.test_number, 1))
        except SelfException.DataTypeConstraint as e:
            logger.exception('MultinomialEM SelfException.DataTypeConstraint')
            e.recall_traceback(sys.exc_info())

        except BaseException:
            logger.exception('MultinomialEM ')
            raise

    def train(self):
        """Training model"""
        # init theta zero
        if self.theta_zero is not None:
            self.prior_pr = self.theta_zero[0]
            self.word_pr = self.theta_zero[1]
        else:
            model = MultinomialNB(self.data)
            model.train()
            self.prior_pr = model.prior_pr
            self.word_pr = model.word_pr

        l = self.data.train_labeled_number
        u = self.data.train_unlabeled_number
        c = self.data.class_number
        d = self.data.feature_number

        # NOTE DELTA and mle are Decimal type
        # EM algorithm
        loop_count = 0

        # delta_0 estimate
        delta = [[Decimal(0) for j in range(c)] for i in range(l+u)]
        # labeled data
        for i in range(l):
            delta[i][int(self.data.train_yl[i])] = Decimal(1)
        # unlabeled data
        for i in range(u):
            temp_sum = Decimal(0)  # should be sum by hand, np.sum return 0
            for j in range(c):
                delta[l+i][j] = Decimal(self.prior_pr[j]) * Utility.multinomial(self.data.train_xu[i], self.word_pr[j])
                temp_sum += delta[l+i][j]
            for j in range(c):
                delta[l+i][j] /= temp_sum

        # MLE_0 calculation
        # Labeled MLE = P(D_L|theta) = sum(i =1 -> l){ log(P(y_i|theta) * P(x_i|y_i, theta)) }
        log_mle_new = Decimal(0)
        log_mle_old = Decimal(0)

        for i in range(l):
            log_mle_new += Decimal(Decimal(self.prior_pr[int(self.data.train_yl[i])]) *
                                  Utility.multinomial(self.data.train_xl[i],
                                                      self.word_pr[int(self.data.train_yl[i])])).ln()
        # unlabeled MLE = P(D_U|theta) =
        # sum(i =1 -> u){ sum(j=1 -> c){ delta[i, j] * log(P(y_i|theta) * P(x_i|y_i, theta)) }}
        for i in range(u):
            for j in range(c):
                log_mle_new += delta[i+l][j] * (Decimal(self.prior_pr[j]) *
                                                Utility.multinomial(self.data.train_xu[i], self.word_pr[j])).ln()

        # data D = (xl, yl) union (xu)
        # the loop continues from theta_1
        # The process starts with M-step first, then E-step for estimating delta with same theta version
        # the same version of (theta, delta) take easier for tracking convergence estimate
        # (which is computed by the same (theta, delta))
        while abs(log_mle_new - log_mle_old) > self.epsilon:
            logger.info('Diff: ' + str(log_mle_new - log_mle_old))
            loop_count += 1
            # M step
            self.prior_pr = np.zeros(c)
            self.word_pr = np.zeros((c, d))

            # theta_t-1 estimate
            # add-one smoothing in use
            for j in range(c):
                for i in range(l):
                    self.prior_pr[j] += float(delta[i][j])
                    self.word_pr[j] = self.word_pr[j] + float(delta[i][j]) * self.data.train_xl[i]
                for i in range(u):
                    self.prior_pr[j] += float(delta[i+l][j])
                    self.word_pr[j] = self.word_pr[j] + float(delta[i+l][j]) * self.data.train_xu[i]
                #  class prior probability
                self.prior_pr[j] = (self.prior_pr[j]+ 1) / float(l+u+c)

            #  word conditional probability
            sum_word_pr = self.word_pr.sum(axis=1)
            self.word_pr[:] += 1
            self.word_pr = (self.word_pr.T / (sum_word_pr + self.data.feature_number)).T

            # E step
            # delta estimate
            delta = [[Decimal(0) for j in range(c)] for i in range(l+u)]
            for i in range(l):
                delta[i][int(self.data.train_yl[i])] = Decimal(1)
            for i in range(u):
                temp_sum = Decimal(0)
                for j in range(c):
                    delta[i+l][j] = Decimal(self.prior_pr[j]) * Utility.multinomial(self.data.train_xu[i], self.word_pr[j])
                    temp_sum += delta[i+l][j]
                for j in range(c):
                    delta[l+i][j] /= temp_sum

            # check convergence condition
            log_mle_old = log_mle_new
            # MLE calculation
            log_mle_new = Decimal(0)
            for i in range(l):
                log_mle_new += (Decimal(self.prior_pr[int(self.data.train_yl[i])]) *
                                      Utility.multinomial(self.data.train_xl[i],
                                                          self.word_pr[int(self.data.train_yl[i])])).ln()
            for i in range(u):
                for j in range(c):
                    log_mle_new += delta[i+l][j] * (Decimal(self.prior_pr[j]) *
                                                  Utility.multinomial(self.data.train_xu[i], self.word_pr[j])).ln()

        self.EM_loop_count = loop_count

    def test(self):
        """Estimated value of x for each class"""
        estimate_value = [[Decimal(0) for j in range(self.data.class_number)] for i in range(self.data.test_number)]
        for i in range(self.data.test_number):
            max_val = 0
            max_id = -1
            for j in range(self.data.class_number):
                estimate_value[i][j] = Decimal(self.prior_pr[j]) * \
                                       Utility.multinomial(self.data.test_x[i, :], self.word_pr[j, :])
                if max_val < estimate_value[i][j]:
                    max_val = estimate_value[i][j]
                    max_id = j
            self.predicted_label[i, 0] = max_id


class MultinomialManyToOneNB(object):
    __doc__ = 'Deploy all labeled data Naive Bayes using many-to-one assumption for text classification; ' \
              'MLE derivative solution; ' \
              'This only work with SslDataset data (using train_xl).'

    def __init__(self, dataset, component_count_list, component_assignment_list=None):
        try:
            if type(dataset) is not SslDataset: raise SelfException.DataTypeConstraint('Dataset type is not SslDataset')
            self.data = dataset

            # component count list
            if len(component_count_list) != self.data.class_number:
                raise SelfException.MismatchLengthComponentList(
                    'Component list has different length with number of classes')
            if type(component_count_list) is not np.ndarray:
                raise SelfException.ComponentCountIsList(
                    'Component count must be a np.ndarray')
            self.component_count_list = component_count_list
            self.component_number = component_count_list.sum()
            self.component_count_cumulative = np.zeros(self.data.class_number + 1).astype(int)
            self.component_assignment_list = component_assignment_list

            # parameters set
            #  class prior probability [P(y1) ... P(yc)]
            self.prior_pr = np.zeros(self.component_number)
            #  word conditional probability per class [ [P(wi|y1)] ... [P(wi|yc)] ], i=1..d
            self.word_pr = np.zeros((self.component_number, self.data.feature_number))

            # predicted label
            # note: matrix type here, in the same type with data.test_y
            self.predicted_label = np.zeros((self.data.test_number, 1))

        except SelfException.DataTypeConstraint as e:
            logger.exception('MultinomialAllLabeled SelfException.DataTypeConstraint')
            e.recall_traceback(sys.exc_info())

        except BaseException:
            logger.exception('MultinomialAllLabeled BaseException')
            raise

    def train(self):
        """Training model"""
        # init constants
        l = self.data.train_labeled_number
        c = self.data.class_number
        d = self.data.feature_number
        m = self.component_number

        self.component_count_cumulative[0] = 0
        for i in range(1, c + 1):
            self.component_count_cumulative[i] = self.component_count_cumulative[i - 1] + \
                                                 self.component_count_list[i - 1]

        # init delta
        delta = [[Decimal(0) for j in range(m)] for i in range(l)]
        for i in range(l):
            label = int(self.data.train_yl[i])
            if self.component_assignment_list is None:
                # random sampling
                sampling = Utility.equal_sampling(self.component_count_list[label])
                for j in range(self.component_count_cumulative[label], self.component_count_cumulative[label + 1]):
                    delta[i][j] = Decimal(sampling[j - self.component_count_cumulative[label]])
            else:
                # sample from prior assigned component
                for class_id, class_component in enumerate(self.component_assignment_list):
                    for component_id, component_list in enumerate(class_component):
                        for data in component_list:
                            component_location = self.component_count_cumulative[class_id] + component_id
                            delta[data][component_location] = Decimal(1)

        # estimate
        for i in range(l):
            label = int(self.data.train_yl[i])
            for j in range(self.component_count_cumulative[label], self.component_count_cumulative[label + 1]):
                self.prior_pr[j] += float(delta[i][j])
                self.word_pr[j] = self.word_pr[j] + float(delta[i][j]) * self.data.train_xl[i]
        # add-one smoothing in use
        #  class prior probability
        self.prior_pr[:] += 1
        self.prior_pr = np.divide(self.prior_pr, float(self.data.train_labeled_number + m))
        #  word conditional probability
        sum_word_pr = self.word_pr.sum(axis=1)
        self.word_pr[:] += 1
        self.word_pr = (self.word_pr.T / (sum_word_pr + self.data.feature_number)).T

    def test(self):
        """Estimated value of x for each class"""
        estimate_value = [[Decimal(0) for j in range(self.data.class_number)] for i in range(self.data.test_number)]
        for i in range(self.data.test_number):
            max_val = 0
            max_id = -1
            for j in range(self.data.class_number):
                for k in range(self.component_count_cumulative[j], self.component_count_cumulative[j+1]):
                    estimate_value[i][j] += Decimal(self.prior_pr[k]) * \
                                            Utility.multinomial(self.data.test_x[i], self.word_pr[k])
                if max_val < estimate_value[i][j]:
                    max_val = estimate_value[i][j]
                    max_id = j
            self.predicted_label[i, 0] = max_id


class MultinomialManyToOneEM(object):
    __doc__ = 'Deploy Semi-supervised Multinomial model for text classification; ' \
              'many to one assumption, EM algorithm; ' \
              'This only work with SslDataset data (using train_xl).'

    def __init__(self, dataset, component_count_list, component_assignment_list=None, epsilon=1e-4):
        """
        init the data for model.
        component_assignment_list contains the data init for each component.
        In case of None component_assignment,
        the random partial probability for all components in a class will be derived.
        :param dataset: data
        :param component_count_list: list or 1-d array, list number of components for each class
        :param component_assignment_list: 3-d ndarray or list of list of list,
                                          component assignment list for each labeled data.
                                          If this list is None, random sampling will be used.
        :param epsilon: default 1e-4, MLE convergence threshold
        """
        try:
            if type(dataset) is not SslDataset:
                raise SelfException.DataTypeConstraint('Dataset type is not SslDataset')
            self.data = dataset
            # component count list
            if len(component_count_list) != self.data.class_number:
                raise SelfException.MismatchLengthComponentList(
                    'Component list has different length with number of classes')
            if type(component_count_list) is not np.ndarray:
                raise SelfException.ComponentCountIsList(
                    'Component count must be a np.ndarray')
            self.component_count_list = component_count_list
            self.component_number = component_count_list.sum()
            self.component_count_cumulative = np.zeros(self.data.class_number + 1).astype(int)
            self.component_assignment_list = component_assignment_list

            # parameters set
            #  component prior probability [P(m1) ... P(mc)]
            self.prior_pr = np.zeros(self.component_number)
            #  word conditional probability per component [ [P(wi|m1)] ... [P(wi|m...)] ], i=1..d
            self.word_pr = np.zeros((self.component_number, self.data.feature_number))
            self.EM_loop_count = -1
            self.epsilon = epsilon

            # predicted label
            # note: matrix type here, in the same type with data.test_y
            self.predicted_label = np.zeros((self.data.test_number, 1))

        except SelfException.DataTypeConstraint as e:
            logger.exception('MultinomialManyToOne SelfException.DataTypeConstraint')
            e.recall_traceback(sys.exc_info())

        except BaseException:
            logger.exception('MultinomialManyToOne BaseException')
            raise

    def train(self):
        """Training model"""
        # init constants
        l = self.data.train_labeled_number
        u = self.data.train_unlabeled_number
        c = self.data.class_number
        d = self.data.feature_number
        m = self.component_number

        self.component_count_cumulative[0] = 0
        for i in range(1, c+1):
            self.component_count_cumulative[i] = self.component_count_cumulative[i-1] + self.component_count_list[i-1]

        # init delta
        delta = [[Decimal(0) for j in range(m)] for i in range(l+u)]
        for i in range(l):
            label = int(self.data.train_yl[i])
            if self.component_assignment_list is None:
                # random sampling
                sampling = Utility.equal_sampling(self.component_count_list[label])
                for j in range(self.component_count_cumulative[label], self.component_count_cumulative[label+1]):
                    delta[i][j] = Decimal(sampling[j-self.component_count_cumulative[label]])
            else:
                # sample from prior assigned component
                for class_id, class_component in enumerate(self.component_assignment_list):
                    for component_id, component_list in enumerate(class_component):
                        for data in component_list:
                            component_location = self.component_count_cumulative[class_id] + component_id
                            delta[data][component_location] = Decimal(1)

        # theta_0 estimate
        for i in range(l):
            label = int(self.data.train_yl[i])
            for j in range(self.component_count_cumulative[label], self.component_count_cumulative[label + 1]):
                self.prior_pr[j] += float(delta[i][j])
                self.word_pr[j] = self.word_pr[j] + float(delta[i][j]) * self.data.train_xl[i]
        # add-one smoothing in use
        #  class prior probability
        self.prior_pr[:] += 1
        self.prior_pr = np.divide(self.prior_pr, float(self.data.train_labeled_number + m))
        #  word conditional probability
        sum_word_pr = self.word_pr.sum(axis=1)
        self.word_pr[:] += 1
        self.word_pr = (self.word_pr.T/(sum_word_pr + self.data.feature_number)).T

        # EM algorithm
        loop_count = 0

        # delta_0 estimate
        delta = [[Decimal(0) for j in range(m)] for i in range(l+u)]
        for i in range(l):
            temp_sum = Decimal(0)
            for j in range(m):
                delta[i][j] = Decimal(self.prior_pr[j]) * Utility.multinomial(self.data.train_xl[i], self.word_pr[j])
                temp_sum += delta[i][j]
            for j in range(m):
                delta[i][j] = delta[i][j] / temp_sum

        for i in range(u):
            temp_sum = Decimal(0)
            for j in range(m):
                delta[l+i][j] = Decimal(self.prior_pr[j]) * Utility.multinomial(self.data.train_xu[i], self.word_pr[j])
                temp_sum += delta[l+i][j]
            for j in range(m):
                delta[l+i][j] = delta[l+i][j] / temp_sum

        # MLE_0 calculation
        log_mle_new = Decimal(0)
        log_mle_old = Decimal(0)
        for j in range(m):
            for i in range(l):
                if float(delta[i][j]) != 0:
                    log_mle_new += delta[i][j] * \
                                   Decimal(Decimal(self.prior_pr[j]) *
                                           Utility.multinomial(self.data.train_xl[i], self.word_pr[j])).ln()
            for i in range(u):
                if float(delta[l+i][j]) != 0:
                    log_mle_new += delta[l+i][j] * \
                                   Decimal(Decimal(self.prior_pr[j]) *
                                           Utility.multinomial(self.data.train_xu[i], self.word_pr[j])).ln()

        # data D = (xl, yl) union (xu)
        # the loop continues from theta_1
        # The process is started with M-step first, then E-step for estimating delta with same theta version
        # the same version of (theta, delta) take easier for tracking convergence estimate
        # (which is computed by the same (theta, delta))
        while abs(log_mle_new - log_mle_old) > self.epsilon:
            logger.info('Diff: ' + str(log_mle_new - log_mle_old))
            loop_count += 1
            # M step
            self.prior_pr = np.zeros(m)
            self.word_pr = np.zeros((m, d))
            # normalize delta with component constraint of labeled data
            for i in range(l):
                label = int(self.data.train_yl[i])
                for j in range(self.component_count_cumulative[label]):
                    delta[i][j] = Decimal(0)
                for j in range(self.component_count_cumulative[label+1], m):
                    delta[i][j] = Decimal(0)
                # re-normalize delta of component for each label sum to 1
                temp_sum = sum(delta[i])
                for j in range(self.component_count_cumulative[label], self.component_count_cumulative[label+1]):
                    delta[i][j] /= temp_sum

            # add-one smoothing in use
            for j in range(m):
                for i in range(l):
                    self.prior_pr[j] += float(delta[i][j])
                    self.word_pr[j] = self.word_pr[j] + float(delta[i][j]) * self.data.train_xl[i]
                for i in range(u):
                    self.prior_pr[j] += float(delta[i+l][j])
                    self.word_pr[j] = self.word_pr[j] + float(delta[i+l][j]) * self.data.train_xu[i]
                #  class prior probability
                self.prior_pr[j] = (self.prior_pr[j] + 1) / float(l+u+m)

            #  word conditional probability
            sum_word_pr = self.word_pr.sum(axis=1)
            self.word_pr[:] += 1
            self.word_pr = (self.word_pr.T / (sum_word_pr + self.data.feature_number)).T

            # E step
            # delta estimate
            delta = [[ Decimal(0) for j in range(m)] for i in range(l+u)]
            for i in range(l):
                temp_sum = Decimal(0)
                for j in range(m):
                    delta[i][j] = Decimal(self.prior_pr[j]) * \
                                  Utility.multinomial(self.data.train_xl[i], self.word_pr[j])
                    temp_sum += delta[i][j]
                for j in range(m):
                    delta[i][j] /= temp_sum

            for i in range(u):
                temp_sum = Decimal(0)
                for j in range(m):
                    delta[l+i][j] = Decimal(self.prior_pr[j]) * \
                                    Utility.multinomial(self.data.train_xu[i], self.word_pr[j])
                    temp_sum += delta[l+i][j]
                for j in range(m):
                    delta[l+i][j] /= temp_sum

            # check convergence condition
            log_mle_old = log_mle_new
            # MLE calculation
            log_mle_new = Decimal(0)
            for j in range(m):
                for i in range(l):
                    if float(delta[i][j]) != 0:
                        log_mle_new += delta[i][j] * \
                                       Decimal(Decimal(self.prior_pr[j]) *
                                               Utility.multinomial(self.data.train_xl[i], self.word_pr[j])).ln()
                for i in range(u):
                    if float(delta[l+i][j] != 0):
                        log_mle_new += delta[l+i][j] * \
                                   Decimal(Decimal(self.prior_pr[j]) *
                                           Utility.multinomial(self.data.train_xu[i], self.word_pr[j])).ln()

        self.EM_loop_count = loop_count

    def test(self):
        """Estimated value of x for each class"""
        estimate_value = [[Decimal(0) for j in range(self.data.class_number)] for i in range(self.data.test_number)]
        for i in range(self.data.test_number):
            max_val = 0
            max_id = -1
            for j in range(self.data.class_number):
                for k in range(self.component_count_cumulative[j], self.component_count_cumulative[j+1]):
                    estimate_value[i][j] += Decimal(self.prior_pr[k]) * \
                                            Utility.multinomial(self.data.test_x[i], self.word_pr[k])
                if max_val < estimate_value[i][j]:
                    max_val = estimate_value[i][j]
                    max_id = j
            self.predicted_label[i, 0] = max_id


""" Hierarchy tree data type.
Despite saving the mean vector, it is easier to save sum of all vector and sum of sum vector.
When we merge tree, the updating only sums both of them.
:param sum_vector: vector sum of count
:param element_id_list: list of tree's element id
:param splitter_list: location and order of splitter use for re-building the tree
"""
hierarchy_tree = namedlist(
    'hierarchy_tree', [('sum_vector', None), ('element_id_list',[]), ('splitter_list',[])])

""" Splitter
a splitter including index in element_id_list and order number
:param id: index of splitter in element_id_list
:param order: oder of splitter counted from 0
"""
splitter = namedlist('splitter', 'cut_id, cut_value, order', default=0)

class AgglomerativeTree(object):
    __doc__ = 'Agglomerative Hierarchy Tree'

    def __init__(self, dataset, metric='match_distance'):
        try:
            if type(dataset) is not SslDataset:
                raise SelfException.DataTypeConstraint('Dataset type is not SslDataset.')
            self.data = dataset

            if metric == 'bin_bin_distance':
                self.metric = self.bin_bin_distance
            elif metric == 'match_distance':
                self.metric = self.match_distance
            else:
                raise SelfException.NonexistentMetric('Distance metric is nonexistent.')

        except SelfException.DataTypeConstraint as e:
            logger.exception('AgglomerativeTree SelfException.DataTypeConstraint')
            e.recall_traceback(sys.exc_info())

        except BaseException:
            logger.exception('AgglomerativeTree BasseException')
            raise

    @staticmethod
    def bin_bin_distance(x, y):
        """
        bin-to-bin metric using Chi-square
        :param x: hierarchy_tree type
        :param y: hierarchy_tree type
        :return: distance between x and y
        """
        try:
            if type(x) is not hierarchy_tree or type(y) is not hierarchy_tree:
                raise SelfException.DataTypeConstraint('bin_bin_distance: input type must be hierarchy_tree')
            x_mean = x.sum_vector / float(len(x.element_id_list))
            y_mean = y.sum_vector / float(len(y.element_id_list))
            return (np.square(x_mean - y_mean) / (2 * (x_mean + y_mean))).sum()

        except SelfException.DataTypeConstraint as e:
            logger.exception('bin_bin_distance SelfException.DataTypeConstraint')
            e.recall_traceback(sys.exc_info())

        except BaseException:
            logger.exception('bin_bin_distance')
            raise

    @staticmethod
    def match_distance(x, y):
        """
        match metric using Chi-square
        :param x: hierarchy_tree type
        :param y: hierarchy_tree type
        :return: distance between x and y
        """
        try:
            if type(x) is not hierarchy_tree and type(y) is not hierarchy_tree:
                raise SelfException.DataTypeConstraint('bin_bin_distance: input type must be hierarchy_tree')
            x_mean = x.sum_vector / float(len(x.element_id_list))
            y_mean = y.sum_vector / float(len(y.element_id_list))
            # cumulative vector
            for i in range(1, len(x_mean)):
                x_mean[i] += x_mean[i-1]
                y_mean[i] += y_mean[i - 1]
            return np.absolute(x_mean - y_mean).sum()

        except SelfException.DataTypeConstraint as e:
            logger.exception('match_distance SelfException.DataTypeConstraint')
            e.recall_traceback(sys.exc_info())

        except BaseException:
            logger.exception('match_distance')
            raise

    def build_hierarchy_tree(self, data_list):
        """
        build agglomerative tree
        * The algorithm is simple the merge of hierarchy_tree and the split of each layer is split_list.
        * Each time two hierarchy_tree are merged, the two split_list are merged too.
        * Also the index list splitter.index of the right side merged hierarchy_tree will be updated
        * by number of elements of the right one. The splitter.order does not need to update.
        :param data_list: list id of labeled data
        :return: hierarchy_tree of data_list
        """
        try:
            if type(data_list) is not list:
                raise SelfException.DataTypeConstraint('data_list must be a list.')
            if not all([type(i) is hierarchy_tree for i in data_list]):
                raise SelfException.DataTypeConstraint('data_list must be a list of hierarchy_tree.')
            # init metric matrix
            cluster_metric = np.full((len(data_list), len(data_list)), np.inf)
            for i in range(len(data_list)):
                for j in range(i + 1, len(data_list)):
                    cluster_metric[i, j] = self.metric(data_list[i], data_list[j])
                    cluster_metric[j, i] = cluster_metric[i, j]

            splitter_count = -1
            while len(data_list) > 1:
                splitter_count += 1
                min_index = np.unravel_index(cluster_metric.argmin(), np.shape(cluster_metric))
                # merge 2 trees min_index[0] = [ min_index[0], min_index[1] ]
                # the order does not importance there.
                data_list[min_index[0]].sum_vector += data_list[min_index[1]].sum_vector
                # update splitter index of min_index[1] when cluster min_index[0] is merge on its left side
                extend_length = len(data_list[min_index[0]].element_id_list)
                for split in data_list[min_index[1]].splitter_list:
                    split.cut_id += extend_length
                data_list[min_index[0]].element_id_list.extend(data_list[min_index[1]].element_id_list)
                data_list[min_index[0]].splitter_list.extend(data_list[min_index[1]].splitter_list)
                # set new splitter between 2 new merged clusters
                data_list[min_index[0]].splitter_list.append(splitter(cut_id=extend_length-1,
                                                                      cut_value=cluster_metric[min_index],
                                                                      order=splitter_count))

                # TODO: Finding a fast way to update metric matrix
                # update metric matrix for new min_index[0]
                for i in range(len(data_list)):
                    if i != min_index[0]:
                        cluster_metric[min_index[0], i] = self.metric(data_list[min_index[0]], data_list[i])
                        cluster_metric[i, min_index[0]] = cluster_metric[min_index[0], i]

                # delete min_index[1] after merged
                # It is better to delete later, we do not mess the new merged cluster id in cluster_metric
                del data_list[min_index[1]]
                cluster_metric = np.delete(cluster_metric, min_index[1], axis=0)
                cluster_metric = np.delete(cluster_metric, min_index[1], axis=1)
            return data_list[0]

        except SelfException.DataTypeConstraint as e:
            logger.exception('build_hierarchy_tree SelfException.DataTypeConstraint')
            e.recall_traceback(sys.exc_info())

        except BaseException:
            logger.exception('build_hierarchy_tree')
            raise

    def build_hierarchy_scheme_multi_class(self):
        x = self.data.train_xl
        y = self.data.train_yl
        hierarchy_scheme = []
        data_group_by_label = [[] for _ in range(self.data.class_number)]

        # group data by label
        # first init each data point as a tree
        for counter, val in enumerate(y):
            data_group_by_label[val].append(
                hierarchy_tree(sum_vector=x[counter], element_id_list=[counter], splitter_list=[]))

        # build tree for each group data by class
        for tree in data_group_by_label:
            hierarchy_scheme.append(self.build_hierarchy_tree(tree))
        return hierarchy_scheme

    def build_hierarchy_scheme_binary_class(self):
        # NOTICE that this only for specific case of binary class when we only assert tree on the label '1'
        # Only build tree on negative class label '1'
        # the returned hierarchy_scheme contains
        X = self.data.train_xl
        y = self.data.train_yl

        # assign empty hierarchy_tree (contain all data label '0') for label '0'
        lable0_tree = hierarchy_tree(sum_vector=[], element_id_list=[], splitter_list=[])
        data_group_label1 = []

        # group data by label
        # For label 0: only collect all data point has lable '0'
        # For label 1: first init each data point as a tree
        for counter, val in enumerate(y):
            if val == 1:
                data_group_label1.append(
                    hierarchy_tree(sum_vector=X[counter], element_id_list=[counter], splitter_list=[]))
            else:
                lable0_tree.element_id_list.append(counter)

        # build tree for each group data by class
        hierarchy_scheme = [lable0_tree, self.build_hierarchy_tree(data_group_label1)]
        return hierarchy_scheme


class NewsEvaluation(object):
    __doc__ = 'Evaluation methods using 20news dataset' \
              'train number 11269' \
              'test number 7505' \
              'vocabulary size 61188'
    result_return_form = namedlist('result_return_form', 'accuracy, precision, recall, f1, support')

    def __init__(self):
        self.default_dir = 'data/'
        self.map_filename, self.train_filename, self.test_filename = 'news.map.csv', 'news.train.csv', 'news.test.csv'

        # exp_feature_selection_1a
        self.sub_folder_list_1a = '1a_scale 1a_no_scale'.split()

        # exp_cooperation_unlabeled_1b
        self.sub_folder_list_1b = '1b_scale 1b_no_scale'.split()
        self.approximate_labeled_sizes_1b = np.array([100, 200, 500, 700, 1000, 1500, 2000, 2500, 3000, 4000])

        # exp_feature_selection_1c
        self.sub_folder_list_1c = '1c_no_scale 1c_scale_l1 1c_scale_l2'.split()

    def report_export(self, model, file_name, extend_file=False, detail_return=False):
        """
        Export report
        :param model:
        :param file_name: file name
        :param extend_file: bool, set true if extend train and test files while export result
        :return:
        """
        label = model.data.class_name_list
        target = model.data.test_y
        prediction = model.predicted_label

        # calculate evaluation index
        accuracy = metrics.accuracy_score(target, prediction)
        report = metrics.classification_report(target, prediction, target_names=label)

        # export report
        if extend_file:
            with open(file_name, 'a') as f:
                f.write(report)
                f.writelines('\n#labeled: ' + str(model.data.train_labeled_number))
                f.writelines('\n#unlabeled: ' + str(model.data.train_unlabeled_number))
                f.writelines('\nAccuracy: ' + str(accuracy))
                f.writelines('\n')
        else:
            with open(file_name, 'w') as f:
                f.write(report)
                f.writelines('\n#labeled: ' + str(model.data.train_labeled_number))
                f.writelines('\n#unlabeled: ' + str(model.data.train_unlabeled_number))
                f.writelines('\nAccuracy: ' + str(accuracy))
                f.writelines('\n')

        # return value in detail
        if detail_return:
            detail = metrics.precision_recall_fscore_support(target, prediction)
            return self.result_return_form(accuracy=accuracy,
                                           precision=detail[0].round(4), recall=detail[1].round(4),
                                           f1=detail[2].round(4), support=detail[3])

    def report_avg_report(self, file_name, title, result):
        """
        export avg(summary) result (accuracy, precision, recall, f1)
        The export file will be extend to write from the end
        :param file_name: string, file to export
        :param title: string, title to print
        :param result: result_return_form, result to export
        :return:
        """
        with open(file_name, 'a') as f:
            f.writelines('\n' + title)
            f.writelines('\n Accuracy ' + str(result.accuracy))
            f.writelines('\n Precision ' + str(result.precision))
            f.writelines('\n Recall ' + str(result.recall))
            f.writelines('\n f1 ' + str(result.f1))
            f.writelines('\n support ' + str(result.support))
            f.writelines('\n')


    # I. The advantage of unlabeled data
    # a) test feature selection
    def exp_feature_selection_1a(self, unlabeled_size=5000, n_splits=5, random_seed=0, epsilon=1e-4):
        """
        exp the feature selection. Experiment proceduce:
        1. Scaling data
        2. Feature selection using Mutual Information (MI) word rank

        Exp Model
        NB and EM with 2 types of data: scaling and non-scaling
        in corresponding with difference number of selected word features.

        Expected:
        - The upper performance of scaling data
        - Finding a good range of word number should be chosen

        Data Reading:
        The func will scan default_dir location and process through all sub-folder in sub_folder_list (one-by-one).
        In each sub-folder contains all test cases for one exp.
        The process will perform the algorithm and return the result file in the same folder of each test case.

        :param unlabeled_size: size of unlabeled training data
        :param n_splits: number of split fold for train labeled data
        :param random_seed: default = 0, random seed
        :param epsilon: default 1e-4, MLE convergence threshold for EM based algorithm
        :return:
        """
        logger.info('Start Evaluation - exp_feature_selection_1a')
        logger.info('unlabeled_size: ' + str(unlabeled_size))
        logger.info('n_splits: ' + str(n_splits))
        sub_folder_list = self.sub_folder_list_1a
        nb_result_filename = 'NB_1a_result.log'
        em_result_filename = 'EM_1a_result.log'
        try:
            # this default exp uses unlabeled_size data as unlabeled data,
            # the remaining is split into n_splits non-overlap parts with size 1000
            for sub_folder in sub_folder_list:
                # get all tests in sub-folder
                test_folder_list = next(os.walk(self.default_dir + sub_folder + '/'))[1]
                for test_folder in test_folder_list:
                    test_dir = self.default_dir + sub_folder + '/' + test_folder + '/'
                    logger.info('START TEST: ' + test_dir)
                    origin_data = Dataset()
                    origin_data.load_from_csv([test_dir + self.map_filename,
                                        test_dir + self.train_filename, test_dir + self.test_filename])
                    origin_ssl_data = SslDataset(origin_data, unlabeled_size=unlabeled_size, random_seed=random_seed)
                    skf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

                    # split train data into 5 overlap parts and get the average
                    avg_NB_result = None
                    avg_EM_result = None
                    for _, testcase_train_index in skf.split(origin_ssl_data.train_xl, origin_ssl_data.train_yl):
                        # TODO Check the copied elements
                        testcase_data = SslDataset(origin_ssl_data)
                        testcase_data.train_xl = origin_ssl_data.train_xl[testcase_train_index]
                        testcase_data.train_yl = origin_ssl_data.train_yl[testcase_train_index]
                        testcase_data.train_labeled_number = len(testcase_train_index)

                        # Test Naive Bayes
                        logger.info('START: Naive Bayes')
                        nb_model = MultinomialNB(testcase_data)
                        nb_model.train()
                        nb_model.test()
                        temp_result = self.report_export(nb_model, test_dir + nb_result_filename,
                                                         extend_file=True, detail_return=True)
                        if avg_NB_result is None:
                            avg_NB_result = temp_result
                        else:
                            avg_NB_result.accuracy += temp_result.accuracy
                            avg_NB_result.precision += temp_result.precision
                            avg_NB_result.recall += temp_result.recall
                            avg_NB_result.f1 += temp_result.f1
                            avg_NB_result.support += temp_result.support
                        logger.info('DONE: Naive Bayes')

                        # Test EM
                        logger.info('START: EM')
                        em_model = MultinomialEM(testcase_data, epsilon=epsilon)
                        em_model.train()
                        em_model.test()
                        temp_result = self.report_export(em_model, test_dir + em_result_filename,
                                                         extend_file=True, detail_return= True)
                        if avg_EM_result is None:
                            avg_EM_result = temp_result
                        else:
                            avg_EM_result.accuracy += temp_result.accuracy
                            avg_EM_result.precision += temp_result.precision
                            avg_EM_result.recall += temp_result.recall
                            avg_EM_result.f1 += temp_result.f1
                            avg_EM_result.support += temp_result.support
                        logger.info('DONE: EM')
                        logger.info('Loop count: ' + str(em_model.EM_loop_count))
                    # compute average values
                    avg_NB_result.accuracy, avg_NB_result.precision, \
                    avg_NB_result.recall, avg_NB_result.f1, avg_NB_result.support = \
                        np.divide(avg_NB_result.accuracy, n_splits), \
                        np.divide(avg_NB_result.precision, n_splits), \
                        np.divide(avg_NB_result.recall, n_splits), \
                        np.divide(avg_NB_result.f1, n_splits), \
                        np.divide(avg_NB_result.support, n_splits)
                    avg_EM_result.accuracy, avg_EM_result.precision, \
                    avg_EM_result.recall, avg_EM_result.f1, avg_EM_result.support = \
                        np.divide(avg_EM_result.accuracy, n_splits), \
                        np.divide(avg_EM_result.precision, n_splits), \
                        np.divide(avg_EM_result.recall, n_splits), \
                        np.divide(avg_EM_result.f1, n_splits), \
                        np.divide(avg_EM_result.support, n_splits)
                    self.report_avg_report(test_dir + nb_result_filename, 'AVERAGE NB', avg_NB_result)
                    self.report_avg_report(test_dir + em_result_filename, 'AVERAGE EM', avg_EM_result)

        except BaseException:
            logger.exception('exp_feature_selection_1a BaseException')
            raise

    def exp_cooperate_unlabeled_1b(self, unlabeled_size=5000, n_tries=5, random_seed=0, epsilon=1e-4):
        """
        Experiment proceduce:
        1. Test with fix large amount of unlabeled data, vary types of labeled size
        The size of labeled size is defined by number of fold split of train data (after split unlabeled data)

        Exp Model
        NB and EM

        Expected:
        - The advantage of unlabeled data

        Data Reading:
        The func will scan default_dir location and process through all sub-folder in sub_folder_list (one-by-one).
        In each sub-folder contains all test cases for one exp.
        The process will perform the algorithm and return the result file in the same folder of each test case.
        :param unlabeled_size: int, default=5000, size of unlabeled data
        :param n_tries: int, default=5, number of re-train times
        :param random_seed: int, default=0, seed of random generator
        :param epsilon: default 1e-4, MLE convergence threshold for EM based algorithm
        :return:
        """
        logger.info('Start Evaluation - exp_cooperation_unlabeled_1b')
        logger.info('unlabeled_size: ' + str(unlabeled_size))
        sub_folder_list = self.sub_folder_list_1b
        nb_result_filename = 'NB_1b_result.log'
        em_result_filename = 'EM_1b_result.log'

        try:
            # this default exp uses unlabeled_size data as unlabeled data,
            # the remaining is split into vary non-overlap parts with vary sizes
            for sub_folder in sub_folder_list:
                # get all tests in sub-folder
                test_folder_list = next(os.walk(self.default_dir + sub_folder + '/'))[1]
                for test_folder in test_folder_list:
                    test_dir = self.default_dir + sub_folder + '/' + test_folder + '/'
                    logger.info('START TEST: ' + test_dir)
                    origin_data = Dataset()
                    origin_data.load_from_csv([test_dir + self.map_filename,
                                               test_dir + self.train_filename, test_dir + self.test_filename])
                    origin_ssl_data = SslDataset(origin_data, unlabeled_size=unlabeled_size, random_seed=random_seed)

                    for sub_train_number in self.approximate_labeled_sizes_1b:
                        n_splits = round(len(origin_ssl_data.train_xl) / float(sub_train_number))
                        skf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

                        # split train data into 5 overlap parts and get the average
                        avg_NB_result = None
                        avg_EM_result = None
                        nb_sub_result_filename = str(sub_train_number) + nb_result_filename
                        em_sub_result_filename = str(sub_train_number) + em_result_filename

                        loop_count = 0
                        for _, testcase_train_index in skf.split(origin_ssl_data.train_xl, origin_ssl_data.train_yl):
                            # if n_splits < n_tries (the number of folds is not enough) then the loop run is smaller
                            if loop_count >= n_tries: break

                            # TODO Check the copied elements
                            testcase_data = SslDataset(origin_ssl_data)
                            testcase_data.train_xl = origin_ssl_data.train_xl[testcase_train_index]
                            testcase_data.train_yl = origin_ssl_data.train_yl[testcase_train_index]
                            testcase_data.train_labeled_number = len(testcase_train_index)

                            # Test Naive Bayes
                            logger.info('START: Naive Bayes')
                            nb_model = MultinomialNB(testcase_data)
                            nb_model.train()
                            nb_model.test()
                            temp_result = self.report_export(nb_model, test_dir + nb_sub_result_filename,
                                                             extend_file=True, detail_return=True)
                            if avg_NB_result is None:
                                avg_NB_result = temp_result
                            else:
                                avg_NB_result.accuracy += temp_result.accuracy
                                avg_NB_result.precision += temp_result.precision
                                avg_NB_result.recall += temp_result.recall
                                avg_NB_result.f1 += temp_result.f1
                                avg_NB_result.support += temp_result.support
                            logger.info('DONE: Naive Bayes')

                            # Test EM
                            logger.info('START: EM')
                            em_model = MultinomialEM(testcase_data, epsilon=epsilon)
                            em_model.train()
                            em_model.test()
                            temp_result = self.report_export(em_model, test_dir + em_sub_result_filename,
                                                             extend_file=True, detail_return=True)
                            if avg_EM_result is None:
                                avg_EM_result = temp_result
                            else:
                                avg_EM_result.accuracy += temp_result.accuracy
                                avg_EM_result.precision += temp_result.precision
                                avg_EM_result.recall += temp_result.recall
                                avg_EM_result.f1 += temp_result.f1
                                avg_EM_result.support += temp_result.support
                            logger.info('DONE: EM')
                            logger.info('Loop count: ' + str(em_model.EM_loop_count))

                            loop_count += 1

                        # compute average values
                        avg_NB_result.accuracy, avg_NB_result.precision, \
                        avg_NB_result.recall, avg_NB_result.f1, avg_NB_result.support = \
                            np.divide(avg_NB_result.accuracy, loop_count), \
                            np.divide(avg_NB_result.precision, loop_count), \
                            np.divide(avg_NB_result.recall, loop_count), \
                            np.divide(avg_NB_result.f1, loop_count), \
                            np.divide(avg_NB_result.support, loop_count)

                        avg_EM_result.accuracy, avg_EM_result.precision, \
                        avg_EM_result.recall, avg_EM_result.f1, avg_EM_result.support = \
                            np.divide(avg_EM_result.accuracy, loop_count), \
                            np.divide(avg_EM_result.precision, loop_count), \
                            np.divide(avg_EM_result.recall, loop_count), \
                            np.divide(avg_EM_result.f1, loop_count), \
                            np.divide(avg_EM_result.support, loop_count)
                        self.report_avg_report(test_dir + nb_sub_result_filename, 'AVERAGE NB', avg_NB_result)
                        self.report_avg_report(test_dir + em_sub_result_filename, 'AVERAGE EM', avg_EM_result)

        except BaseException:
            logger.exception('exp_cooperate_unlabeled_1b BaseException')
            raise

    def exp_feature_selection_1c(self, n_splits=5, random_seed=0):
        """
        Exp Model
        NB and EM with 2 types of data: scaling l1 and l2 and non-scaling
        in corresponding with difference number of selected word features.

        Data Reading:
        The func will scan default_dir location and process through all sub-folder in sub_folder_list (one-by-one).
        In each sub-folder contains all test cases for one exp.
        The process will perform the algorithm and return the result file in the same folder of each test case.

        :param n_splits: number of split fold for train labeled data
        :param random_seed: default = 0, random seed
        :return:
        """
        logger.info('Start Evaluation - exp_feature_selection_1c - all labeled data')
        logger.info('n_splits: ' + str(n_splits))
        sub_folder_list = self.sub_folder_list_1c
        nb_result_filename = 'NB_1c_result.log'
        try:
            # this default exp uses unlabeled_size data as unlabeled data,
            # the remaining is split into n_splits non-overlap parts with size 1000
            for sub_folder in sub_folder_list:
                # get all tests in sub-folder
                test_folder_list = next(os.walk(self.default_dir + sub_folder + '/'))[1]
                for test_folder in test_folder_list:
                    test_dir = self.default_dir + sub_folder + '/' + test_folder + '/'
                    logger.info('START TEST: ' + test_dir)
                    origin_data = Dataset()
                    origin_data.load_from_csv([test_dir + self.map_filename,
                                        test_dir + self.train_filename, test_dir + self.test_filename])
                    skf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

                    # split train data into 5 overlap parts and get the average
                    avg_NB_result = None
                    for _, testcase_train_index in skf.split(origin_data.train_x, origin_data.train_y):
                        # TODO Check the copied elements
                        testcase_data = Dataset(origin_data)
                        testcase_data.train_x = origin_data.train_x[testcase_train_index]
                        testcase_data.train_y = origin_data.train_y[testcase_train_index]
                        testcase_data.train_labeled_number = len(testcase_train_index)

                        # Test Naive Bayes
                        logger.info('START: Naive Bayes')
                        nb_model = MultinomialNB(testcase_data)
                        nb_model.train()
                        nb_model.test()
                        temp_result = self.report_export(nb_model, test_dir + nb_result_filename,
                                                         extend_file=True, detail_return=True)
                        if avg_NB_result is None:
                            avg_NB_result = temp_result
                        else:
                            avg_NB_result.accuracy += temp_result.accuracy
                            avg_NB_result.precision += temp_result.precision
                            avg_NB_result.recall += temp_result.recall
                            avg_NB_result.f1 += temp_result.f1
                            avg_NB_result.support += temp_result.support
                        logger.info('DONE: Naive Bayes')
                    # compute average values
                    avg_NB_result.accuracy, avg_NB_result.precision, \
                    avg_NB_result.recall, avg_NB_result.f1, avg_NB_result.support = \
                        np.divide(avg_NB_result.accuracy, n_splits), \
                        np.divide(avg_NB_result.precision, n_splits), \
                        np.divide(avg_NB_result.recall, n_splits), \
                        np.divide(avg_NB_result.f1, n_splits), \
                        np.divide(avg_NB_result.support, n_splits)
                    self.report_avg_report(test_dir + nb_result_filename, 'AVERAGE NB', avg_NB_result)

        except BaseException:
            logger.exception('exp_feature_selection_1c BaseException')
            raise


class BinaryManytoOneEvaluation(object):
    __doc__ = 'Evaluation methods using Reuters21578 dataset' \
              'Data 11367' \
              'vocabulary size 28438'
    result_return_form = namedlist('result_return_form', 'accuracy, precision, recall, f1, support')

    def __init__(self, component_search_threshold=50, component_estimate_nfold=5, max_tries_parameter_estimate=5,
                 distance_metric='match_distance', component_search_epsilon=1e-4, component_estimate_f1_avg='binary'):
        """
        :param component_search_threshold: maximum number of groups search when estimate papameter
        :param component_estimate_nfold: positive int, default=5,
                                            number of fold use for parameter estimate(group speading) at each try

        :param max_tries_parameter_estimate: int, default 5, maximum number of tries run when estimate parameter
        :param distance_metric: type of distance method using for building tree
        :param component_search_epsilon: default 1e-4, MLE convergence EM algorithm threshold for component search
        :param component_estimate_f1_avg: default 'binary', f1 average mode,
                                          'binary': only label 1, 'macro': unweighted mean
        """
        self.default_dir = 'data/'
        self.map_filename, self.train_filename, self.test_filename = 'news.map.csv', 'news.train.csv', 'news.test.csv'

        # exp_feature_selection_2a
        self.sub_folder_list_2a = '2a_reuters_test_scale_3'.split()
        self.approximate_labeled_sizes_2a = np.array([100, 200, 500, 1000])

        # component estimate parameter
        self.component_search_threshold = component_search_threshold
        self.component_estimate_nfold = component_estimate_nfold
        self.max_tries_parameter_estimate = max_tries_parameter_estimate
        self.distance_metric = distance_metric
        self.component_search_epsilon = component_search_epsilon
        # TODO check average mode only macro or binary
        self.component_estimate_f1_avg = component_estimate_f1_avg

    def report_export(self, model, file_name, extend_file=False, detail_return=False):
        """
        Export report
        :param model:
        :param file_name: file name
        :param extend_file: bool, set true if extend train and test files while export result
        :return:
        """
        label = model.data.class_name_list
        target = model.data.test_y
        prediction = model.predicted_label

        # calculate evaluation index
        accuracy = metrics.accuracy_score(target, prediction)
        report = metrics.classification_report(target, prediction, target_names=label)

        # export report
        if extend_file:
            with open(file_name, 'a') as f:
                f.write(report)
                f.writelines('\n#labeled: ' + str(model.data.train_labeled_number))
                f.writelines('\n#unlabeled: ' + str(model.data.train_unlabeled_number))
                f.writelines('\nAccuracy: ' + str(accuracy))
                f.writelines('\n')
        else:
            with open(file_name, 'w') as f:
                f.write(report)
                f.writelines('\n#labeled: ' + str(model.data.train_labeled_number))
                f.writelines('\n#unlabeled: ' + str(model.data.train_unlabeled_number))
                f.writelines('\nAccuracy: ' + str(accuracy))
                f.writelines('\n')

        # return value in detail
        if detail_return:
            detail = metrics.precision_recall_fscore_support(target, prediction)
            return self.result_return_form(accuracy=accuracy,
                                           precision=detail[0].round(4), recall=detail[1].round(4),
                                           f1=detail[2].round(4), support=detail[3])

    def report_avg_report(self, file_name, title, result):
        """
        export avg(summary) result (accuracy, precision, recall, f1)
        The export file will be extend to write from the end
        :param file_name: string, file to export
        :param title: string, title to print
        :param result: result_return_form, result to export
        :return:
        """
        with open(file_name, 'a') as f:
            f.writelines('\n' + title)
            f.writelines('\n Accuracy ' + str(result.accuracy))
            f.writelines('\n Precision ' + str(result.precision))
            f.writelines('\n Recall ' + str(result.recall))
            f.writelines('\n f1 ' + str(result.f1))
            f.writelines('\n support ' + str(result.support))
            f.writelines('\n')

    def estimated_random_many_one_component(self, dataset, selected_model='EM', max_try=5):
        """
        Estimated number of component per class using random sampling
        Currently this search only uses for Reuters binary classification with default class lable '0'.
        Then it only searhc component for the second label '1'.
        :param dataset:
        :param selected_model: 'NB' for Naive Bayes or 'EM' for EM algorithm
        :param max_try: default =1, maximum try for each cross-validation (Not recommended, this forces the calc times)
        :return:
        """
        logger.info('START estimated_random_many_one_component')
        if dataset.class_number > 2:
            raise SelfException.UnSupportMethod('Only supports binary classification')
        try:
            selected_parameter = np.asarray([1, 1])
            skf = model_selection.StratifiedKFold(n_splits=self.component_estimate_nfold, shuffle=True, random_state=0)
            max_f1 = -1
            best_component = 1
            for i in range(2, self.component_search_threshold):
                selected_parameter[1] = i
                avg_f1 = 0
                try_count = 0
                for train_index, test_index in skf.split(dataset.train_xl, dataset.train_yl):
                    try_count += 1
                    if try_count > max_try: break
                    # generate testcase
                    testcase_data = SslDataset(dataset)
                    testcase_data.train_xl = dataset.train_xl[train_index]
                    testcase_data.train_yl = dataset.train_yl[train_index]
                    testcase_data.train_labeled_number = len(train_index)
                    testcase_data.test_x = dataset.train_xl[test_index]
                    testcase_data.test_y = dataset.train_yl[test_index]
                    testcase_data.test_number = len(test_index)
                    if selected_model == 'EM':
                        model = MultinomialManyToOneEM(testcase_data,
                                                       component_count_list=selected_parameter,
                                                       epsilon=self.component_search_epsilon)
                    elif selected_model == 'NB':
                        model = MultinomialManyToOneNB(testcase_data,
                                                       component_count_list=selected_parameter)
                    else:
                        raise SelfException.UnSupportMethod('Only NB or EM option is accepted')
                    model.train()
                    model.test()
                    avg_f1 += metrics.f1_score(model.data.test_y, model.predicted_label,
                                               average=self.component_estimate_f1_avg)
                if avg_f1 > max_f1:
                    max_f1 = avg_f1
                    best_component = i
            selected_parameter[1] = best_component
            logger.info('Best random selected component number: ' + str(best_component))
            return selected_parameter
        except BaseException:
            logger.exception('estimated_random_many_one_component BaseException')
            raise

    def estimated_tree_cut_value_many_one_component(self, dataset, selected_model='EM', max_try=5):
        """
        Estimated number of component per class using aglomarative hierarchy tree sampling cut value.
        For each fold, function will finds the best cut value in range of component_threshold

        Currently this search only uses for Reuters binary classification with default class label '0'.
        Then it only search component for the second label '1'.


        The CV procedure in search to find the best cut_value with highest f1_score on all folds
        :param dataset:
        :param selected_model: 'NB' for Naive Bayes or 'EM' for EM algorithm
        :param max_try: default =1, maximum try for each cross-validation (Not recommended, this forces the calc times)
        :return:
        """
        # TODO Estimate with no split (only 1 group)  case
        logger.info('START estimated_tree_cut_value_many_one_component')
        if dataset.class_number > 2:
            raise SelfException.UnSupportMethod('Only supports binary classification')
        try:
            try_count = 0
            avg_split_value = 0
            skf = model_selection.StratifiedKFold(n_splits=self.component_estimate_nfold, shuffle=True, random_state=0)
            for train_index, test_index in skf.split(dataset.train_xl, dataset.train_yl):
                try_count += 1
                if try_count > max_try: break
                # generate testcase
                testcase_data = SslDataset(dataset)
                testcase_data.train_xl = dataset.train_xl[train_index]
                testcase_data.train_yl = dataset.train_yl[train_index]
                testcase_data.train_labeled_number = len(train_index)
                testcase_data.test_x = dataset.train_xl[test_index]
                testcase_data.test_y = dataset.train_yl[test_index]
                testcase_data.test_number = len(test_index)

                # build hierarchy tree
                # NOTICE that this only for specific case of binary class when we only assert tree on label '1'
                tree = AgglomerativeTree(testcase_data, metric=self.distance_metric)
                hierarchy_scheme = tree.build_hierarchy_scheme_binary_class()
                # get list id data for each class
                component_assignment_label0 = hierarchy_scheme[0].element_id_list
                component_assignment_label1 = hierarchy_scheme[1].element_id_list
                # get the splitters to reconstruct the tree
                # the reconstruction starts from splitter has the last order (highest order value)
                splitter_scheme_label1 = sorted(hierarchy_scheme[1].splitter_list, key=lambda x: x.order, reverse=True)

                # start to reconstruct tree
                extract_count = 0
                split_id_list = []
                max_f1_score = 0
                best_split_value = 0
                for split in splitter_scheme_label1:
                    if extract_count >= self.component_search_threshold: break
                    extract_count += 1
                    split_id_list.append(split.cut_id)
                    # Slice component_list by split_id_list
                    component_assignment_label1_sliced = []
                    slice_range = np.insert(sorted(split_id_list), len(split_id_list),
                                            len(component_assignment_label1)-1)
                    u = 0
                    for v in slice_range:
                        component_assignment_label1_sliced.append(component_assignment_label1[u:v+1])
                        u = v + 1

                    selected_component_count = np.asarray([1, len(component_assignment_label1_sliced)])
                    selected_component_assignment = [[component_assignment_label0], component_assignment_label1_sliced]
                    # initialize model
                    if selected_model == 'EM':
                        model = MultinomialManyToOneEM(testcase_data,
                                                       component_count_list=selected_component_count,
                                                       component_assignment_list=selected_component_assignment,
                                                       epsilon=self.component_search_epsilon)
                    elif selected_model == 'NB':
                        model = MultinomialManyToOneNB(testcase_data,
                                                       component_count_list=selected_component_count,
                                                       component_assignment_list=selected_component_assignment)
                    else:
                        raise SelfException.UnSupportMethod('Only NB or EM option is accepted')
                    model.train()
                    model.test()
                    temp_f1 = metrics.f1_score(model.data.test_y, model.predicted_label,
                                               average=self.component_estimate_f1_avg)
                    if temp_f1 > max_f1_score:
                        max_f1_score = temp_f1
                        best_split_value = split.cut_value
                # sum up best_split_value after each try
                avg_split_value += best_split_value

            # based on what condition comes first: run through all fold or try count passes max_try
            avg_split_value = avg_split_value / min(self.component_estimate_nfold, max_try)
            logger.info('Average Split Value: ' + str(avg_split_value))

            # After find out avg_split_value, build tree on all training set
            # and find the cut with smallest absolute distance witht avg_split_value
            tree = AgglomerativeTree(dataset, metric=self.distance_metric)
            hierarchy_scheme = tree.build_hierarchy_scheme_binary_class()
            # get list id data for each class
            component_assignment_label0 = hierarchy_scheme[0].element_id_list
            component_assignment_label1 = hierarchy_scheme[1].element_id_list
            # get the splitters and find the closest split with avg_split_value
            split_id_list = []
            # finding the upper bound
            for split in sorted(hierarchy_scheme[1].splitter_list, key=lambda x: x.order, reverse=True):
                split_id_list.append(split.cut_id)
                if split.cut_value > avg_split_value: break

            # split data by split_id_list
            component_assignment_label1_sliced = []
            slice_range = np.insert(sorted(split_id_list), len(split_id_list),
                                    len(component_assignment_label1) - 1)
            u = 0
            for v in slice_range:
                component_assignment_label1_sliced.append(component_assignment_label1[u:v + 1])
                u = v + 1
            # assign parameter
            selected_component_count = np.asarray([1, len(component_assignment_label1)])
            selected_component_assignment = [[component_assignment_label0], component_assignment_label1_sliced]
            logger.info('Number of component: ' + str(len(component_assignment_label1_sliced)))
            return selected_component_count, selected_component_assignment
        except BaseException:
            logger.exception('estimated_tree_cut_value_many_one_component BaseException')
            raise

    def estimated_tree_component_count_many_one_component(self, dataset, selected_model='EM', max_try=5):
        """
        Estimated number of component per class using aglomarative hierarchy tree sampling cut value.
        For each fold, function will finds the best component number has highest f1 in range of component_threshold

        Currently this search only uses for Reuters binary classification with default class label '0'.
        Then it only search component for the second label '1'.


        The CV procedure in search to find the best cut_value with highest f1_score on all folds
        :param dataset:
        :param selected_model: 'NB' for Naive Bayes or 'EM' for EM algorithm
        :param max_try: default =1, maximum try for each cross-validation (Not recommended, this forces the calc times)
        :return:
        """
        # TODO Estimate with no split (only 1 group)  case
        logger.info('START estimated_tree_component_count_many_one_component')
        if dataset.class_number > 2:
            raise SelfException.UnSupportMethod('Only supports binary classification')
        try:
            max_f1 = -1
            best_component = 1
            skf = model_selection.StratifiedKFold(n_splits=self.component_estimate_nfold, shuffle=True, random_state=0)
            for component_number in range(2, self.component_search_threshold):
                avg_f1 = 0
                try_count = 0
                for train_index, test_index in skf.split(dataset.train_xl, dataset.train_yl):
                    try_count += 1
                    if try_count > max_try: break
                    # generate testcase
                    testcase_data = SslDataset(dataset)
                    testcase_data.train_xl = dataset.train_xl[train_index]
                    testcase_data.train_yl = dataset.train_yl[train_index]
                    testcase_data.train_labeled_number = len(train_index)
                    testcase_data.test_x = dataset.train_xl[test_index]
                    testcase_data.test_y = dataset.train_yl[test_index]
                    testcase_data.test_number = len(test_index)

                    # build hierarchy tree
                    # NOTICE that this only for specific case of binary class when we only assert tree on label '1'
                    tree = AgglomerativeTree(testcase_data, metric=self.distance_metric)
                    hierarchy_scheme = tree.build_hierarchy_scheme_binary_class()
                    # get list id data for each class
                    component_assignment_label0 = hierarchy_scheme[0].element_id_list
                    component_assignment_label1 = hierarchy_scheme[1].element_id_list
                    # get the splitters to reconstruct the tree
                    # the reconstruction starts from splitter has the last order (highest order value)
                    splitter_scheme_label1 = sorted(hierarchy_scheme[1].splitter_list, key=lambda x: x.order, reverse=True)

                    # start to reconstruct tree into specific number of components
                    extract_count = 0
                    split_id_list = []
                    # create split id list
                    for split in splitter_scheme_label1:
                        # if data has less than component_number instances. The most component will be extracted
                        extract_count += 1
                        if extract_count > component_number: break
                        split_id_list.append(split.cut_id)

                    # Slice component_list by split_id_list
                    slice_range = np.insert(sorted(split_id_list), len(split_id_list),
                                            len(component_assignment_label1)-1)
                    u = 0
                    component_assignment_label1_sliced = []
                    for v in slice_range:
                        component_assignment_label1_sliced.append(component_assignment_label1[u:v+1])
                        u = v + 1
                    selected_component_count = np.asarray([1, len(component_assignment_label1_sliced)])
                    selected_component_assignment = [[component_assignment_label0], component_assignment_label1_sliced]
                    # initialize model
                    if selected_model == 'EM':
                        model = MultinomialManyToOneEM(testcase_data,
                                                       component_count_list=selected_component_count,
                                                       component_assignment_list=selected_component_assignment,
                                                       epsilon=self.component_search_epsilon)
                    elif selected_model == 'NB':
                        model = MultinomialManyToOneNB(testcase_data,
                                                       component_count_list=selected_component_count,
                                                       component_assignment_list=selected_component_assignment)
                    else:
                        raise SelfException.UnSupportMethod('Only NB or EM option is accepted')
                    model.train()
                    model.test()
                    avg_f1 += metrics.f1_score(model.data.test_y, model.predicted_label,
                                               average=self.component_estimate_f1_avg)
                if avg_f1 > max_f1:
                    max_f1 = avg_f1
                    best_component = component_number

            # After estimated best_component, build tree on all training set
            # and find the component split for best_component number
            tree = AgglomerativeTree(dataset, metric=self.distance_metric)
            hierarchy_scheme = tree.build_hierarchy_scheme_binary_class()
            # get list id data for each class
            component_assignment_label0 = hierarchy_scheme[0].element_id_list
            component_assignment_label1 = hierarchy_scheme[1].element_id_list
            # get the splitters and find the closest split with avg_split_value
            split_id_list = []
            extract_count = 0

            # finding the upper bound
            for split in sorted(hierarchy_scheme[1].splitter_list, key=lambda x: x.order, reverse=True):
                extract_count += 1
                if extract_count > best_component: break
                split_id_list.append(split.cut_id)

            # split data by split_id_list
            slice_range = np.insert(sorted(split_id_list), len(split_id_list),
                                    len(component_assignment_label1) - 1)
            component_assignment_label1_sliced = []
            u = 0
            for v in slice_range:
                component_assignment_label1_sliced.append(component_assignment_label1[u:v + 1])
                u = v + 1

            # assign parameter
            selected_component_count = np.asarray([1, len(component_assignment_label1)])
            selected_component_assignment = [[component_assignment_label0], component_assignment_label1_sliced]
            logger.info('Number of component tree selected: ' + str(len(component_assignment_label1_sliced)))
            return selected_component_count, selected_component_assignment
        except BaseException:
            logger.exception('estimated_tree_component_count_many_one_component BaseException')
            raise

    # II. Data grouping assumption
    #2a using Reuteters dataset
    def exp_group_assumption_2a(self, unlabeled_size=5000, n_tries=5, random_seed=0,
                                epsilon=1e-4, tree_search='cut_value'):
        """
        Experiment proceduce:
        1. Test with fix large amount of unlabeled data, with small amount of labeled data
        At each try, parameter will be selected from cross-validation using n_parameter_estimate_tries folds

        Exp Model
        Only EM original and NB with 3 different assumption:
        - (1) Origin
        - (2) Group with random assignment
        - (3) Group with hierarchy assignment

        Expected:
        - The advantage of many-to-one assumption.

        Data Reading:
        The func will scan default_dir location and process through all sub-folder in sub_folder_list (one-by-one).
        In each sub-folder contains all test cases for one exp.
        The process will perform the algorithm and return the result file in the same folder of each test case.
        :param unlabeled_size: int, default=5000, size of unlabeled data
        :param n_tries: int, default=5, number of re-train times
        :param random_seed: int, default=0, seed of random generator
        :param epsilon: default 1e-4, MLE convergence threshold for EM based algorithm
        :param tree_search: 'cut_value' or 'component_count',
                            default 'cut_value', search method for tree component assignment
        :return:
        """
        logger.info('Start Evaluation - exp_group_assumption_2a')
        logger.info('unlabeled_size: ' + str(unlabeled_size))
        sub_folder_list = self.sub_folder_list_2a
        if tree_search not in ['cut_value', 'component_count']:
            raise SelfException.UnSupportMethod('tree_search only accept cut_value or component_count')

        # FIXME Add more algorithm here
        nb1_result_filename = 'NB1_2a_result.log'
        nb2_result_filename = 'NB2_2a_result.log'
        nb3_result_filename = 'NB3_2a_result.log'
        em1_result_filename = 'EM1_2a_result.log'
        em2_result_filename = 'EM2_2a_result.log'
        em3_result_filename = 'EM3_2a_result.log'

        try:
            for sub_folder in sub_folder_list:
                # get all tests in sub-folder
                test_folder_list = next(os.walk(self.default_dir + sub_folder + '/'))[1]
                for test_folder in test_folder_list:
                    test_dir = self.default_dir + sub_folder + '/' + test_folder + '/'
                    logger.info('START TEST: ' + test_dir)
                    origin_data = Dataset()
                    origin_data.load_from_csv([test_dir + self.map_filename,
                                               test_dir + self.train_filename, test_dir + self.test_filename])
                    origin_ssl_data = SslDataset(origin_data, unlabeled_size=unlabeled_size, random_seed=random_seed)

                    for sub_train_number in self.approximate_labeled_sizes_2a:
                        n_splits = round(len(origin_ssl_data.train_xl) / float(sub_train_number))
                        skf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

                        # split train data into 5 overlap parts and get the average
                        # FIXME Add more algorithm here
                        avg_NB1_result = None
                        avg_NB2_result = None
                        avg_NB3_result = None
                        avg_EM1_result = None
                        avg_EM2_result = None
                        avg_EM3_result = None
                        # FIXME Add more algorithm here
                        nb1_sub_result_filename = str(sub_train_number) + nb1_result_filename
                        nb2_sub_result_filename = str(sub_train_number) + nb2_result_filename
                        nb3_sub_result_filename = str(sub_train_number) + nb3_result_filename
                        em1_sub_result_filename = str(sub_train_number) + em1_result_filename
                        em2_sub_result_filename = str(sub_train_number) + em2_result_filename
                        em3_sub_result_filename = str(sub_train_number) + em3_result_filename

                        loop_count = 0
                        for _, testcase_train_index in skf.split(origin_ssl_data.train_xl, origin_ssl_data.train_yl):
                            # if n_splits < n_tries (the number of folds is not enough) then the loop run is smaller
                            if loop_count >= n_tries: break

                            # TODO Check the copied elements
                            testcase_data = SslDataset(origin_ssl_data)
                            testcase_data.train_xl = origin_ssl_data.train_xl[testcase_train_index]
                            testcase_data.train_yl = origin_ssl_data.train_yl[testcase_train_index]
                            testcase_data.train_labeled_number = len(testcase_train_index)

                            # FIXME Add more algorithm here
                            # Test NB 1
                            logger.info('START: Naive Bayes 1 (origin)')
                            nb1_model = MultinomialNB(testcase_data)
                            nb1_model.train()
                            nb1_model.test()
                            temp_result = self.report_export(nb1_model, test_dir + nb1_sub_result_filename,
                                                             extend_file=True, detail_return=True)
                            if avg_NB1_result is None:
                                avg_NB1_result = temp_result
                            else:
                                avg_NB1_result.accuracy += temp_result.accuracy
                                avg_NB1_result.precision += temp_result.precision
                                avg_NB1_result.recall += temp_result.recall
                                avg_NB1_result.f1 += temp_result.f1
                                avg_NB1_result.support += temp_result.support
                            logger.info('DONE: Naive Bayes 1 (origin)')

                            # Test EM 1
                            logger.info('START: EM 1 (origin)')
                            em1_model = MultinomialEM(testcase_data, epsilon=epsilon)
                            em1_model.train()
                            em1_model.test()
                            temp_result = self.report_export(em1_model, test_dir + em1_sub_result_filename,
                                                             extend_file=True, detail_return=True)
                            if avg_EM1_result is None:
                                avg_EM1_result = temp_result
                            else:
                                avg_EM1_result.accuracy += temp_result.accuracy
                                avg_EM1_result.precision += temp_result.precision
                                avg_EM1_result.recall += temp_result.recall
                                avg_EM1_result.f1 += temp_result.f1
                                avg_EM1_result.support += temp_result.support
                            logger.info('DONE: EM 1 (origin)')
                            logger.info('Loop count: ' + str(em1_model.EM_loop_count))

                            # Test NB 2
                            logger.info('START: Naive Bayes 2 (random many-to-one)')
                            estimated_component = self.estimated_random_many_one_component(
                                testcase_data, selected_model='NB',
                                max_try=self.max_tries_parameter_estimate)
                            nb2_model = MultinomialManyToOneNB(testcase_data, component_count_list=estimated_component)
                            nb2_model.train()
                            nb2_model.test()
                            temp_result = self.report_export(nb2_model, test_dir + nb2_sub_result_filename,
                                                             extend_file=True, detail_return=True)
                            if avg_NB2_result is None:
                                avg_NB2_result = temp_result
                            else:
                                avg_NB2_result.accuracy += temp_result.accuracy
                                avg_NB2_result.precision += temp_result.precision
                                avg_NB2_result.recall += temp_result.recall
                                avg_NB2_result.f1 += temp_result.f1
                                avg_NB2_result.support += temp_result.support
                            logger.info('DONE: Naive Bayes 2 (random many-to-one)')

                            # Test EM 2
                            logger.info('START: EM 2 (random many-to-one)')
                            estimated_component = self.estimated_random_many_one_component(
                                testcase_data, selected_model='EM',
                                max_try=self.max_tries_parameter_estimate)
                            em2_model = MultinomialManyToOneEM(testcase_data, epsilon=epsilon,
                                                               component_count_list=estimated_component)
                            em2_model.train()
                            em2_model.test()
                            temp_result = self.report_export(em2_model, test_dir + em2_sub_result_filename,
                                                             extend_file=True, detail_return=True)
                            if avg_EM2_result is None:
                                avg_EM2_result = temp_result
                            else:
                                avg_EM2_result.accuracy += temp_result.accuracy
                                avg_EM2_result.precision += temp_result.precision
                                avg_EM2_result.recall += temp_result.recall
                                avg_EM2_result.f1 += temp_result.f1
                                avg_EM2_result.support += temp_result.support
                            logger.info('DONE: EM 2 (random many-to-one)')
                            logger.info('Loop count: ' + str(em2_model.EM_loop_count))

                            # Test NB 3
                            logger.info('START: Naive Bayes 3 (hierarchy tree many-to-one)')
                            if tree_search == 'cut_value':
                                estimated_component_count, estimated_component_assignment = \
                                    self.estimated_tree_cut_value_many_one_component(
                                        testcase_data, selected_model='NB',
                                        max_try=self.max_tries_parameter_estimate)
                            else:
                                estimated_component_count, estimated_component_assignment = \
                                    self.estimated_tree_component_count_many_one_component(
                                        testcase_data, selected_model='NB',
                                        max_try=self.max_tries_parameter_estimate)

                            nb3_model = MultinomialManyToOneNB(testcase_data,
                                                               component_count_list=estimated_component_count,
                                                               component_assignment_list=estimated_component_assignment)
                            nb3_model.train()
                            nb3_model.test()
                            temp_result = self.report_export(nb3_model, test_dir + nb3_sub_result_filename,
                                                             extend_file=True, detail_return=True)
                            if avg_NB3_result is None:
                                avg_NB3_result = temp_result
                            else:
                                avg_NB3_result.accuracy += temp_result.accuracy
                                avg_NB3_result.precision += temp_result.precision
                                avg_NB3_result.recall += temp_result.recall
                                avg_NB3_result.f1 += temp_result.f1
                                avg_NB3_result.support += temp_result.support
                            logger.info('DONE: Naive Bayes 3 (hierarchy tree many-to-one)')

                            # Test EM 3
                            logger.info('START: EM 3 (hierarchy tree many-to-one)')
                            if tree_search == 'cut_value':
                                estimated_component_count, estimated_component_assignment = \
                                    self.estimated_tree_cut_value_many_one_component(
                                        testcase_data, selected_model='EM',
                                        max_try=self.max_tries_parameter_estimate)
                            else:
                                estimated_component_count, estimated_component_assignment = \
                                    self.estimated_tree_component_count_many_one_component(
                                        testcase_data, selected_model='EM',
                                        max_try=self.max_tries_parameter_estimate)

                            em3_model = MultinomialManyToOneEM(testcase_data, epsilon=epsilon,
                                                               component_count_list=estimated_component_count,
                                                               component_assignment_list=estimated_component_assignment)
                            em3_model.train()
                            em3_model.test()
                            temp_result = self.report_export(em3_model, test_dir + em3_sub_result_filename,
                                                             extend_file=True, detail_return=True)
                            if avg_EM3_result is None:
                                avg_EM3_result = temp_result
                            else:
                                avg_EM3_result.accuracy += temp_result.accuracy
                                avg_EM3_result.precision += temp_result.precision
                                avg_EM3_result.recall += temp_result.recall
                                avg_EM3_result.f1 += temp_result.f1
                                avg_EM3_result.support += temp_result.support
                            logger.info('DONE: EM 3 (hierarchy tree many-to-one)')
                            logger.info('Loop count: ' + str(em3_model.EM_loop_count))

                            # Test end here
                            loop_count += 1
                        # compute average values
                        # FIXME Add more algorithm here
                        avg_NB1_result.accuracy, avg_NB1_result.precision, \
                        avg_NB1_result.recall, avg_NB1_result.f1, avg_NB1_result.support = \
                            np.divide(avg_NB1_result.accuracy, loop_count), \
                            np.divide(avg_NB1_result.precision, loop_count), \
                            np.divide(avg_NB1_result.recall, loop_count), \
                            np.divide(avg_NB1_result.f1, loop_count), \
                            np.divide(avg_NB1_result.support, loop_count)

                        avg_EM1_result.accuracy, avg_EM1_result.precision, \
                        avg_EM1_result.recall, avg_EM1_result.f1, avg_EM1_result.support = \
                            np.divide(avg_EM1_result.accuracy, loop_count), \
                            np.divide(avg_EM1_result.precision, loop_count), \
                            np.divide(avg_EM1_result.recall, loop_count), \
                            np.divide(avg_EM1_result.f1, loop_count), \
                            np.divide(avg_EM1_result.support, loop_count)

                        avg_NB2_result.accuracy, avg_NB2_result.precision, \
                        avg_NB2_result.recall, avg_NB2_result.f1, avg_NB2_result.support = \
                            np.divide(avg_NB2_result.accuracy, loop_count), \
                            np.divide(avg_NB2_result.precision, loop_count), \
                            np.divide(avg_NB2_result.recall, loop_count), \
                            np.divide(avg_NB2_result.f1, loop_count), \
                            np.divide(avg_NB2_result.support, loop_count)

                        avg_EM2_result.accuracy, avg_EM2_result.precision, \
                        avg_EM2_result.recall, avg_EM2_result.f1, avg_EM2_result.support = \
                            np.divide(avg_EM2_result.accuracy, loop_count), \
                            np.divide(avg_EM2_result.precision, loop_count), \
                            np.divide(avg_EM2_result.recall, loop_count), \
                            np.divide(avg_EM2_result.f1, loop_count), \
                            np.divide(avg_EM2_result.support, loop_count)

                        avg_NB3_result.accuracy, avg_NB3_result.precision, \
                        avg_NB3_result.recall, avg_NB3_result.f1, avg_NB3_result.support = \
                            np.divide(avg_NB3_result.accuracy, loop_count), \
                            np.divide(avg_NB3_result.precision, loop_count), \
                            np.divide(avg_NB3_result.recall, loop_count), \
                            np.divide(avg_NB3_result.f1, loop_count), \
                            np.divide(avg_NB3_result.support, loop_count)

                        avg_EM3_result.accuracy, avg_EM3_result.precision, \
                        avg_EM3_result.recall, avg_EM3_result.f1, avg_EM3_result.support = \
                            np.divide(avg_EM3_result.accuracy, loop_count), \
                            np.divide(avg_EM3_result.precision, loop_count), \
                            np.divide(avg_EM3_result.recall, loop_count), \
                            np.divide(avg_EM3_result.f1, loop_count), \
                            np.divide(avg_EM3_result.support, loop_count)

                        # FIXME Add more algorithm here
                        self.report_avg_report(test_dir + nb1_sub_result_filename, 'AVERAGE NB1', avg_NB1_result)
                        self.report_avg_report(test_dir + nb2_sub_result_filename, 'AVERAGE NB2', avg_NB2_result)
                        self.report_avg_report(test_dir + nb3_sub_result_filename, 'AVERAGE NB3', avg_NB3_result)
                        self.report_avg_report(test_dir + em1_sub_result_filename, 'AVERAGE EM1', avg_EM1_result)
                        self.report_avg_report(test_dir + em2_sub_result_filename, 'AVERAGE EM2', avg_EM2_result)
                        self.report_avg_report(test_dir + em3_sub_result_filename, 'AVERAGE EM3', avg_EM3_result)
        except BaseException:
            logger.exception('exp_group_assumption_2a BaseException')
            raise


def main():
    try:
        # Test I
        evaluation = NewsEvaluation()
        # evaluation.exp_feature_selection_1a(epsilon=1e-3)
        # evaluation.exp_cooperate_unlabeled_1b(epsilon=1e-3)
        evaluation.exp_feature_selection_1c()

        # Test II
        # __init__(self, component_search_threshold = 50, component_estimate_nfold=5, max_tries_parameter_estimate=5,
        #                  distance_metric='match_distance', component_search_epsilon=1e-4,
        #                  component_estimate_f1_avg='binary'):
        # evaluation = BinaryManytoOneEvaluation(component_search_threshold=50, component_estimate_nfold=5,
        #                                     max_tries_parameter_estimate=5, distance_metric='match_distance',
        #                                     component_search_epsilon=1e-1, component_estimate_f1_avg='macro')

        # exp_group_assumption_2a(self, unlabeled_size=5000, n_tries=5, random_seed=0,
        #                                 epsilon=1e-4, tree_search='cut_value')
        # evaluation.exp_group_assumption_2a(unlabeled_size=5000, n_tries=5, epsilon=1e-1, tree_search='component_count')

        logger.info('Done!')
    except BaseException:
        logger.exception('main() BaseException')
        raise


if __name__ == '__main__':
    main()
