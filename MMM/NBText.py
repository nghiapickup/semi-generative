# MLE for Multinomial Mixture model
# supervised and semi-supervised
#
# @nghia n h | Yamada lab
##
# Implemented code here is better than GMM,
# my priority is estimating my work first

import sys
import numpy as np
import collections
from scipy import special
from sklearn import metrics
from sklearn import model_selection
import MMM.exceptionHandle as SelfException
from namedlist import namedlist


class Utility(object):
    __doc__ = 'Utilities'

    def log_factorial(self, x):
        """
        Compute ln of x factorial
        :return ln(x!):
        """
        return special.gammaln(np.array(x) + 1)


    def multinomial(self, x, word_prior):
        """
        Compute multinomial density function
        :param x: word vector
        :param prior: class conditional probability for word vector
        :return:
        """
        n = np.sum(x)
        x, word_prior = np.array(x), np.array(word_prior)
        result = self.log_factorial(n) - np.sum(self.log_factorial(x)) + np.sum(x * np.log(word_prior))
        return np.exp(result)

    def posteriori_estimate(self, x, word_prior):
        """
        return estimated posteriori of x with NB model
        :param x: word vector
        :param word_prior: class conditional probability for word vector
        :return: estimated posteriori (constant omitted)
        """
        x, word_prior = np.array(x), np.array(word_prior)
        result = np.sum(x * np.log(word_prior))
        return np.exp(result)


class Dataset(object):
    __doc__ = 'Basic data frame' \
              'At this time, this class only supports equal length feature vector input'

    def __init__(self, *args):
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
            e.recall_traceback(sys.exc_info())

        except BaseException:
            print('Unknown error!')
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

    def __init__(self, dataset=None, unlabeled_size=.5):
        """
        This only take instance from exist Dataset and split labeled and unlabeled data by scaling size unlabeled_size
        The default scale is .5
        :param dataset: Base data, if dataset = None: set empty class.
        :param unlabeled_size: size of unlabeled data
        """
        super().__init__(dataset)
        # data structure
        self.train_xl = np.empty((0))
        self.train_yl = np.empty((0))
        self.train_xu = np.empty((0))
        self.train_yu = np.empty((0))

        self.train_labeled_number = 0
        self.train_unlabeled_number = 0

        if dataset is not None:
            # split training data by unlabeled_size
            try:
                if (type(unlabeled_size) is not float) or (1.0 <= unlabeled_size < 0.0 ):
                    raise SelfException.DataSizeConstraint('Unlabeled_size must be a float and in range of [0,1).')
                sss = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=unlabeled_size, random_state=0)
                # notice: set random_state is a constant to make sure that the next scaling is the expand of last data set
                for labeled_indices, unlabeled_indices in sss.split(self.train_x, self.train_y):
                    self.train_xl = self.train_x[labeled_indices]
                    self.train_yl = self.train_y[labeled_indices, 0]
                    self.train_xu = self.train_x[unlabeled_indices]
                    self.train_yu = self.train_y[unlabeled_indices, 0]

                self.train_labeled_number = len(self.train_xl)
                self.train_unlabeled_number = len(self.train_xu)

            except SelfException.DataSizeConstraint as e:
                e.recall_traceback(sys.exc_info())

            except BaseException:
                print('Unknown error!')
                raise


class MultinomialAllLabeled(object):
    __doc__ = 'Deploy all labeled data Multinomial model for text classification; ' \
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
            e.recall_traceback(sys.exc_info())

        except BaseException:
            print('Unknown error!')
            raise

    def train(self):
        """Training model"""
        for i in range(self.data.train_labeled_number):
            self.prior_pr[int(self.data.train_yl[i, 0])] += 1
            for k in range(self.data.feature_number):
                self.word_pr[int(self.data.train_yl[i, 0]),k] += self.data.train_xl[i,k]

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
        estimate_value = np.zeros((self.data.test_number, self.data.class_number))
        for i in range(self.data.test_number):
            for j in range(self.data.class_number):
                estimate_value[i, j] = self.prior_pr[j] * Utility.posteriori_estimate(self.data.test_x[i,:], self.word_pr[j,:])
            self.predicted_label[i, 0] = np.argmax(estimate_value[i])
        # temp
        print('Acc', metrics.accuracy_score(self.data.test_y,self.predicted_label))
        print('prior pr')
        print(self.prior_pr)
        print('Word pr')
        print(self.word_pr)


class MultinomialEM(object):
    __doc__ = 'Deploy Semi-supervised Multinomial model for text classification; ' \
              'EM algorithm; ' \
              'This only work with SslDataset data (using train_xl).'

    def __init__(self, dataset, theta_zero=None):
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

            # predicted label
            # note: matrix type here, in the same type with data.test_y
            self.predicted_label = np.zeros((self.data.test_number, 1))
        except SelfException.DataTypeConstraint as e:
            e.recall_traceback(sys.exc_info())

        except BaseException:
            print('Unknown error!')
            raise

    def train(self):
        """Training model"""
        # init theta zero
        if self.theta_zero is not None:
            self.prior_pr = self.theta_zero[0]
            self.word_pr = self.theta_zero[1]
        else:
            model = MultinomialAllLabeled(self.data)
            model.train()
            self.prior_pr = model.prior_pr
            self.word_pr = model.word_pr

        l = self.data.train_labeled_number
        u = self.data.train_unlabeled_number
        c = self.data.class_number
        d = self.data.feature_number

        # EM algorithm
        loop_count = 0
        epsilon = 1e-3

        # delta_0 estimate
        delta = np.zeros((l + u, c))
        for j in range(c):
            for i in range(l):
                delta[i, j] = self.prior_pr[j] * Utility.multinomial(self.data.train_xl[i], self.word_pr[j:])
            for i in range(u):
                delta[i + l, j] = self.prior_pr[j] * Utility.multinomial(self.data.train_xu[i], self.word_pr[j:])
        delta = (delta.T / delta.sum(axis=1)).T

        # MLE_0 calculation
        log_mle_new = 0
        log_mle_old = 0
        for j in range(c):
            for i in range(l):
                log_mle_new += delta[i, j] * np.log(
                    self.prior_pr[j] * Utility.multinomial(self.data.train_xl[i], self.word_pr[j, :]))
            for i in range(u):
                log_mle_new += delta[i, j] * np.log(
                    self.prior_pr[j] * Utility.multinomial(self.data.train_xu[i], self.word_pr[j, :]))

        # data D = (xl, yl) union (xu)
        # the loop continues from theta_1
        # The process is started with M-step first, then E-step for estimating delta with same theta version
        # the same version of (theta, delta) take easier for tracking convergence estimate
        # (which is computed by the same (theta, delta))
        while abs(log_mle_new - log_mle_old) > epsilon:
            loop_count += 1
            # M step
            self.prior_pr = np.zeros(c)
            self.word_pr = np.zeros((c, d))
            # normalize delta for labeled data which is following the theirs true label
            for i in range(l):
                for j in range(c):
                    if self.data.train_yl[i] == j:
                        delta[i, j] = 1
                    else:
                        delta[i,j] = 0
            # add-one smoothing in use
            for j in range(c):
                for i in range(l):
                    self.prior_pr[j] += delta[i, j]
                    for k in range(d):
                        self.word_pr[j, k] += delta[i, j] * self.data.train_xl[i, k]
                for i in range(u):
                    self.prior_pr[j] += delta[i + l, j]
                    for k in range(d):
                        self.word_pr[j, k] += delta[i + l, j] * self.data.train_xu[i, k]
                #  class prior probability
                self.prior_pr[j] = (self.prior_pr[j] + 1) / float(l + u + c)

            #  word conditional probability
            sum_word_pr = self.word_pr.sum(axis=1)
            self.word_pr[:] += 1
            self.word_pr = (self.word_pr.T / (sum_word_pr + self.data.feature_number)).T

            # E step
            # delta estimate
            delta = np.zeros((l + u, c))
            for j in range(c):
                for i in range(l):
                    delta[i, j] = self.prior_pr[j] * Utility.multinomial(self.data.train_xl[i], self.word_pr[j:])
                for i in range(u):
                    delta[i + l, j] = self.prior_pr[j] * Utility.multinomial(self.data.train_xu[i], self.word_pr[j:])
            delta = (delta.T / delta.sum(axis=1)).T

            # check convergence condition
            log_mle_old = log_mle_new
            # MLE calculation
            log_mle_new = 0
            for j in range(c):
                for i in range(l):
                    log_mle_new += delta[i, j] * np.log(
                        self.prior_pr[j] * Utility.multinomial(self.data.train_xl[i], self.word_pr[j, :]))
                for i in range(u):
                    log_mle_new += delta[i, j] * np.log(
                        self.prior_pr[j] * Utility.multinomial(self.data.train_xu[i], self.word_pr[j, :]))

        print('EM loops count: ', loop_count)

    def test(self):
        """Estimated value of x for each class"""
        estimate_value = np.zeros((self.data.test_number, self.data.class_number))
        for i in range(self.data.test_number):
            for j in range(self.data.class_number):
                estimate_value[i, j] = self.prior_pr[j] * Utility.posteriori_estimate(self.data.test_x[i, :], self.word_pr[j, :])
            self.predicted_label[i, 0] = np.argmax(estimate_value[i])
        # temp
        print('acc', metrics.accuracy_score(self.data.test_y, self.predicted_label))
        print('prior pr')
        print(self.prior_pr)
        print('Word pr')
        print(self.word_pr)


class MultinomialManyToOne(object):
    __doc__ = 'Deploy Semi-supervised Multinomial model for text classification; ' \
              'many to one assumption, EM algorithm; ' \
              'This only work with SslDataset data (using train_xl).'

    def __init__(self, dataset, component_count, component_assignment=None):
        """
        init the data for model.
        Component_assignment contains the data init for each component.
        In case of None component_assignment,
        the random partial probability for all components in a class will be derived.
        :param dataset: data
        :param component_count: list number of components for each class
        :param component_assignment: component assignment for each labeled data
        """
        try:
            if type(dataset) is not SslDataset:
                raise SelfException.DataTypeConstraint('Dataset type is not SslDataset')
            self.data = dataset
            # component count list
            if len(component_count) != self.data.class_number:
                raise SelfException.MismatchLengthComponentList(
                    'Component list has different length with number of classes')
            if type(component_count) is not list:
                raise SelfException.ComponentCountIsList(
                    'Component count must be a list')
            self.component_count = component_count
            self.component_number = np.sum(component_count)
            self.component_count_sum = np.zeros(self.data.class_number + 1).astype(int)
            if component_assignment is not None:
                self.component_assignment = component_assignment

            # parameters set
            #  component prior probability [P(m1) ... P(mc)]
            self.prior_pr = np.zeros(self.component_number)
            #  word conditional probability per component [ [P(wi|m1)] ... [P(wi|m...)] ], i=1..d
            self.word_pr = np.zeros((self.component_number, self.data.feature_number))

            # predicted label
            # note: matrix type here, in the same type with data.test_y
            self.predicted_label = np.zeros((self.data.test_number, 1))

        except SelfException.DataTypeConstraint as e:
            e.recall_traceback(sys.exc_info())

        except BaseException:
            print('Unknown error!')
            raise

    def equal_sampling(self, component_number):
        """
        Return a list component_number elements of uniform samplings with constraint sum all element is 1
        :param component_number: number of component for of class
        :return: list of randomly sampling component for one class
        """
        samples = np.random.uniform(0, 1, component_number-1)
        samples = np.append(samples, [0, 1])
        samples.sort()
        for i in range(len(samples) - 1):
            samples[i] = samples[i+1] - samples[i]
        return samples[:-1]

    def train(self):
        """Training model"""
        # init constants
        l = self.data.train_labeled_number
        u = self.data.train_unlabeled_number
        c = self.data.class_number
        d = self.data.feature_number
        m = self.component_number

        self.component_count_sum[0] = 0
        for i in range(1, c + 1):
            self.component_count_sum[i] = self.component_count_sum[i] + self.component_count[i -1]

        # init delta: randomly assign class prior for labeled data
        delta = np.zeros((l + u, m))
        for i in range(l):
            label = int(self.data.train_yl[i])
            if self.component_assignment is None:
                # random sampling
                sampling = self.equal_sampling(self.component_count[label])
                for j in range(self.component_count_sum[label], self.component_count_sum[label + 1]):
                    delta[i,j] = sampling[j - self.component_count_sum[label]]
            else:
                # sample from prior comp def

                # TODO implement

                print('df')

        # theta_0 estimate
        for i in range(l):
            label = int(self.data.train_yl[i])
            for j in range(self.component_count_sum[label], self.component_count_sum[label + 1]):
                self.prior_pr[j] += delta[i, j]
                for k in range(d):
                    self.word_pr[j, k] += self.data.train_xl[i, k] * delta[i, j]
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
        epsilon = 1e-3

        # delta_0 estimate
        delta = np.zeros((l + u, m))
        for j in range(m):
            for i in range(l):
                delta[i, j] = self.prior_pr[j] * Utility.multinomial(self.data.train_xl[i], self.word_pr[j:])
            for i in range(u):
                delta[i + l, j] = self.prior_pr[j] * Utility.multinomial(self.data.train_xu[i], self.word_pr[j:])
        delta = (delta.T / delta.sum(axis=1)).T

        # MLE_0 calculation
        log_mle_new = 0
        log_mle_old = 0
        for j in range(m):
            for i in range(l):
                log_mle_new += delta[i, j] * np.log(
                    self.prior_pr[j] * Utility.multinomial(self.data.train_xl[i], self.word_pr[j, :]))
            for i in range(u):
                log_mle_new += delta[i, j] * np.log(
                    self.prior_pr[j] * Utility.multinomial(self.data.train_xu[i], self.word_pr[j, :]))

        # data D = (xl, yl) union (xu)
        # the loop continues from theta_1
        # The process is started with M-step first, then E-step for estimating delta with same theta version
        # the same version of (theta, delta) take easier for tracking convergence estimate
        # (which is computed by the same (theta, delta))
        while abs(log_mle_new - log_mle_old) > epsilon:
            loop_count += 1
            # M step
            self.prior_pr = np.zeros(m)
            self.word_pr = np.zeros((m, d))
            # normalize delta with component constraint of labeled data
            for i in range(l):
                label = int(self.data.train_yl[i])
                for j in range(self.component_count_sum[label]):
                    delta[i, j] = 0
                for j in range(self.component_count_sum[label + 1], m):
                    delta[i, j] = 0
                # re-normalize delta of component for each label sum to 1
                temp_sum = np.sum(delta[i,:])
                for j in range(self.component_count_sum[label], self.component_count_sum[label + 1]):
                    delta[i, j] /= float(temp_sum)

            # add-one smoothing in use
            for j in range(m):
                for i in range(l):
                    self.prior_pr[j] += delta[i, j]
                    for k in range(d):
                        self.word_pr[j, k] += delta[i, j] * self.data.train_xl[i, k]
                for i in range(u):
                    self.prior_pr[j] += delta[i + l, j]
                    for k in range(d):
                        self.word_pr[j, k] += delta[i + l, j] * self.data.train_xu[i, k]
                #  class prior probability
                self.prior_pr[j] = (self.prior_pr[j] + 1) / float(l + u + m)

            #  word conditional probability
            sum_word_pr = self.word_pr.sum(axis=1)
            self.word_pr[:] += 1
            self.word_pr = (self.word_pr.T / (sum_word_pr + self.data.feature_number)).T

            # E step
            # delta estimate
            delta = np.zeros((l + u, m))
            for j in range(m):
                for i in range(l):
                    delta[i, j] = self.prior_pr[j] * Utility.multinomial(self.data.train_xl[i], self.word_pr[j:])
                for i in range(u):
                    delta[i + l, j] = self.prior_pr[j] * Utility.multinomial(self.data.train_xu[i], self.word_pr[j:])
            delta = (delta.T / delta.sum(axis=1)).T

            # check convergence condition
            log_mle_old = log_mle_new
            # MLE calculation
            log_mle_new = 0
            for j in range(m):
                for i in range(l):
                    log_mle_new += delta[i, j] * np.log(
                        self.prior_pr[j] * Utility.multinomial(self.data.train_xl[i], self.word_pr[j, :]))
                for i in range(u):
                    log_mle_new += delta[i, j] * np.log(
                        self.prior_pr[j] * Utility.multinomial(self.data.train_xu[i], self.word_pr[j, :]))

        print('EM loops count: ', loop_count)

    def test(self):
        """Estimated value of x for each class"""
        estimate_value = np.zeros((self.data.test_number, self.data.class_number))
        for i in range(self.data.test_number):
            for j in range(self.data.class_number):
                for k in range(self.component_count_sum[i], self.component_count_sum[i+1]):
                    estimate_value[i, j] += self.prior_pr[k] * Utility.posteriori_estimate(self.data.test_x[i, :], self.word_pr[k, :])
            self.predicted_label[i, 0] = np.argmax(estimate_value[i])
        # temp
        print('acc', metrics.accuracy_score(self.data.test_y, self.predicted_label))
        print('prior pr')
        print(self.prior_pr)
        print('Word pr')
        print(self.word_pr)


""" Hierarchy tree data type.
Despite saving the mean vector, it is easier to save sum of all vector and sum of sum vector.
When we merge tree, the updating only sum both of them.
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
splitter = namedlist('splitter', 'id, order', default=0)

class AgglomerativeTree(object):
    __doc__ = 'Agglomerative Hierarchy Tree'

    def __init__(self, dataset, metric='bin_bin_distance'):
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
            e.recall_traceback(sys.exc_info())

        except BaseException:
            print('Unknown error!')
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
            e.recall_traceback(sys.exc_info())

        except BaseException:
            print('Unknown error!')
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
            e.recall_traceback(sys.exc_info())

        except BaseException:
            print('Unknown error!')
            raise

    def build_hierarchy_tree(self, data_list):
        """
        build agglomerative tree
        * The algorithm is simple the merge of hierarchy_tree  and the split of each layer is split_list.
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
                    split.id += extend_length
                data_list[min_index[0]].element_id_list.extend(data_list[min_index[1]].element_id_list)
                data_list[min_index[0]].splitter_list.extend(data_list[min_index[1]].splitter_list)
                # set new splitter between 2 new merged clusters
                data_list[min_index[0]].splitter_list.append(splitter(id=extend_length-1, order=splitter_count))

                # TODO: Finding a fast way to update metric matrix
                # update metric matrix on new min_index[0]
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
            e.recall_traceback(sys.exc_info())

        except BaseException:
            print('Unknown error!')
            raise

    def build_hierarchy_scheme(self):
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


class Evaluation(object):
    __doc__ = 'Evaluation methods'

    def __init__(self, dataset):
        try:
            if type(dataset) is not SslDataset:
                raise SelfException.DataTypeConstraint('Dataset type is not SslDataset.')
            self.data = dataset

            self.hierarchy_scheme = []

        except SelfException.DataTypeConstraint as e:
            e.recall_traceback(sys.exc_info())

        except BaseException:
            print('Unknown error!')
            raise


def main():
    try:
        # if len(sys.argv) > 1:
        #     list_file = sys.argv[1:]
        # else:
        #     list_file = input("command: ").split()

        # TEST
        list_file = ''.split()

        # Extract data
        data = Dataset()
        data.load_from_csv(list_file)

        data1 = SslDataset(data, 0.4)

        model = MultinomialAllLabeled(data1)
        # model = MultinomialEM(data1)
        # model = MultinomialManyToOne(data1, [1, 1])
        model.train()
        model.test()

        print('Done!')
    except BaseException:
        raise


if __name__ == '__main__':
    main()
