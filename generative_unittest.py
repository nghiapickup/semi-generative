# Unittest
# supervised and semi-supervised
#
# @nghia n h | Yamada-lab

import sys
import copy
import unittest
import math
import numpy as np
import scipy.stats
from sklearn import metrics
import exceptionHandle as SelfException
import MMM.NBText as nb
import Data.data_preprocessing as data_pre


#
# Data
#
class Preprocessing20NewsTest(unittest.TestCase):

    demo_file_list = data_pre.file_location_list(
        'Data/20news-bydate/test_data/vocabulary.txt', 'Data/20news-bydate/test_data/data.map',
        'Data/20news-bydate/test_data/train.data', 'Data/20news-bydate/test_data/train.label',
        'Data/20news-bydate/test_data/test.data', 'Data/20news-bydate/test_data/test.label',
        'Data/20news-bydate/test_data/final/news.map.csv',
        'Data/20news-bydate/test_data/final/news.train.csv',
        'Data/20news-bydate/test_data/final/news.test.csv',
        'Data/20news-bydate/test_data/final/data_info.txt')
    data_pre.Preprocessing20News.reuters_stop_word_file = 'Data/reuters_wos.txt'
    data_pre.Preprocessing20News.mi_word_rank_file = 'Data/20news-bydate/test_data/final/mi_word_rank.txt'

    @classmethod
    def setUpClass(cls):

        cls.preprocessing = data_pre.Preprocessing20News()
        cls.preprocessing.file_list = cls.demo_file_list
        # extract mi rank file first for  test_news_data_mi_selection_process
        cls.preprocessing.mutual_information_export()

    def test_news_data_basic_process(self):
        """
        Checking: Tokenize 20news data, only stemming, remove stop words and one time occurrence words.
        Expected result: Word id 1, 8 (counted from 1) is ommited by one time occurrence and stop word list
        This test does not check scale length.
        :return:
        """
        # TODO test scale length. May brr test_news_data_mi_selection_process

        loaded_train_expected = np.asarray([[2., 10., 4., 2., 1., 1., 3., 9., 4.],
                                            [4., 0., 0., 0., 0., 1., 0., 0., 4.],
                                            [3., 3., 0., 1., 1., 1., 8., 5., 1.],
                                            [2., 0., 0., 0., 0., 0., 4., 0., 1.],
                                            [8., 8., 0., 0., 0., 1., 6., 1., 2.],
                                            [2., 1., 0., 0., 0., 0., 4., 5., 2.]])
        loaded_test_expected = np.asarray([[0., 1., 0., 0., 0., 0., 0., 1., 0.],
                                           [2., 0., 0., 0., 0., 1., 4., 1., 2.],
                                           [3., 3., 0., 1., 1., 1., 8., 5., 2.],
                                           [2., 3., 0., 0., 1., 0., 0., 0., 0.]])
        # news_data_basic_process(scale_length=-1, extract_to_file=False)
        self.preprocessing.news_data_basic_process(extract_to_file=False)

        # check class number
        self.assertEqual(20, len(self.preprocessing.loaded_map_data))
        # check loaded train
        self.assertTrue((loaded_train_expected == self.preprocessing.loaded_train_data).all())
        # check loaded map
        self.assertTrue((loaded_test_expected == self.preprocessing.loaded_test_data).all())

    def test_mutual_information_export(self):
        """
        This test checks the element lists of MI calculating.
        :return:
        """
        class_pr_expected = np.asarray([0., 1/3, 1/3, 0., 1/3, 0., 0., 0., 0., 0., 0.,
                                        0., 0., 0., 0., 0., 0., 0., 0., 0.])
        word_pr_expected = np.asarray([1/3, 1., 2/3, 1/6, 1/3,
                                       1/3, 2/3, 1/6, 5/6, 2/3])
        mi_rank_list_expected = np.asarray([1.584962500721156, 0.0, 6.624767660309072, 1.584962500721156,
                                            3.3731384929212496, 3.3731384929212496, 6.6247676603090735,
                                            1.584962500721156, 8.088220835496804, 6.624767660309072])
        mi_rank_id_list_expected = np.asarray([8, 6, 9, 2, 5, 4, 7, 3, 0, 1])

        (class_pr, word_pr, word_conditional_class_pr, word_mi_rank) = self.preprocessing.mutual_information_export()

        # test basic calculation first
        self.assertTrue((class_pr_expected == class_pr).all())
        self.assertTrue((word_pr_expected == word_pr).all())
        self.assertTrue((word_mi_rank == mi_rank_list_expected).all())

        # test mi rank list
        mi_rank_id_list = np.loadtxt(self.preprocessing.mi_word_rank_file)
        self.assertTrue((mi_rank_id_list_expected==mi_rank_id_list).all())

    def test_news_data_mi_selection_process(self):
        """
        Reduce vocabulary size using MI rank list

        [train data]
        [[4., 2., 10., 4., 2., 1., 1., 1., 3., 9., 4.],
        [2., 4., 0., 0., 0., 0., 1., 0., 0., 0., 4.],
        [0., 3., 3., 0., 1., 1., 1., 0., 8., 5., 1.],
        [0., 2., 0., 0., 0., 0., 0., 0., 4., 0., 1.],
        [0., 8., 8., 0., 0., 0., 1., 0., 6., 1., 2.],
        [0., 2., 1., 0., 0., 0., 0., 0., 4., 5., 2.]]

        [test data]
        [[0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 2., 0., 0., 0., 0., 1., 0., 4., 1., 2.],
        [0., 3., 3., 0., 1., 1., 1., 0., 8., 5., 2.],
        [3., 2., 3., 0., 0., 1., 0., 1., 0., 0., 0.]]

        [MI rank list]
        [8, 6, 9, 2, 5, 4, 7, 3, 0, 1]


        [selected_word_number = 5, pickup ids [8, 6, 9, 2, 5]]

        train data
        [[ 3., 1., 9. 10., 1., 4.],
        [ 0., 1., 0., 0., 0., 4.],
        [ 8., 1., 5., 3., 1., 1.],
        [ 4., 0., 0., 0., 0., 1.],
        [ 6., 1., 1., 8., 0., 2.],
        [ 4., 0., 5., 1., 0., 2.]]

        test data
        [[0., 0., 1., 1., 0., 0.],
        [4., 1., 1., 0., 0., 2.],
        [8., 1., 5., 3., 1., 2.],
        [0., 0., 0., 3., 1., 0.]]


        [selected_word_number = 5, pickup ids [8, 6, 9, 2, 5], length scale= 7]

        train data
        [[0.875, 0.2916666666666667, 2.625, 2.9166666666666665, 0.2916666666666667, 4.0],
        [0.0, 7.0, 0.0, 0.0, 0.0, 4.0],
        [3.111111111111111, 0.3888888888888889, 1.9444444444444444, 1.1666666666666667, 0.3888888888888889, 1.0],
        [7.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [2.625, 0.4375, 0.4375, 3.5, 0.0, 2.0],
        [2.8, 0.0, 3.5, 0.7, 0.0, 2.0]]

        test data
        [[0.0, 0.0, 3.5, 3.5, 0.0, 0.0],
        [4.666666666666667, 1.1666666666666667, 1.1666666666666667, 0.0, 0.0, 2.0],
        [3.111111111111111, 0.3888888888888889, 1.9444444444444444, 1.1666666666666667, 0.3888888888888889, 2.0],
        [0.0, 0.0, 0.0, 5.25, 1.75, 0.0]]

        :return:
        """
        loaded_train_expected = np.asarray([[3., 1., 9., 10., 1., 4.],
                                            [0., 1., 0., 0., 0., 4.],
                                            [8., 1., 5., 3., 1., 1.],
                                            [4., 0., 0., 0., 0., 1.],
                                            [6., 1., 1., 8., 0., 2.],
                                            [4., 0., 5., 1., 0., 2.]])
        loaded_test_expected = np.asarray([[0., 0., 1., 1., 0., 0.],
                                           [4., 1., 1., 0., 0., 2.],
                                           [8., 1., 5., 3., 1., 2.],
                                           [0., 0., 0., 3., 1., 0.]])

        loaded_train_scale_expected = np.asarray([[0.875, 0.2916666666666667, 2.625, 2.9166666666666665,
                                                   0.2916666666666667, 4.0],
                                                  [0.0, 7.0, 0.0, 0.0, 0.0, 4.0],
                                                  [3.111111111111111, 0.3888888888888889, 1.9444444444444444,
                                                   1.1666666666666667, 0.3888888888888889, 1.0],
                                                  [7.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                                  [2.625, 0.4375, 0.4375, 3.5, 0.0, 2.0],
                                                  [2.8, 0.0, 3.5, 0.7, 0.0, 2.0]])
        loaded_test_scale_expected = np.asarray([[0.0, 0.0, 3.5, 3.5, 0.0, 0.0],
                                                 [4.666666666666667, 1.1666666666666667,
                                                  1.1666666666666667, 0.0, 0.0, 2.0],
                                                 [3.111111111111111, 0.3888888888888889, 1.9444444444444444,
                                                  1.1666666666666667, 0.3888888888888889, 2.0],
                                                 [0.0, 0.0, 0.0, 5.25, 1.75, 0.0]])

        # Test 1
        feature_size = 5
        scale = -1
        self.preprocessing.news_data_mi_selection_process(selected_word_number=feature_size,
                                                          scale_length=scale, extract_to_file=True)
        # check class number
        self.assertEqual(20, len(self.preprocessing.loaded_map_data))
        # check features size
        self.assertEqual(feature_size + 1, np.shape(self.preprocessing.loaded_train_data)[1])
        self.assertEqual(feature_size + 1, np.shape(self.preprocessing.loaded_test_data)[1])
        # check class number
        self.assertEqual(20, len(self.preprocessing.loaded_map_data))

        # check loaded train
        self.assertTrue((loaded_train_expected == self.preprocessing.loaded_train_data).all())
        # check loaded map
        self.assertTrue((loaded_test_expected == self.preprocessing.loaded_test_data).all())

        # Test 2
        feature_size = 5
        scale = 7.
        self.preprocessing.news_data_mi_selection_process(selected_word_number=feature_size,
                                                          scale_length=scale, extract_to_file=True)
        # check sum all features is equal to scale
        sum_list_train = self.preprocessing.loaded_train_data.T[:-1].sum(axis=0)
        sum_list_test = self.preprocessing.loaded_test_data.T[:-1].sum(axis=0)
        self.assertTrue(np.isclose(sum_list_train, scale).all())
        self.assertTrue(np.isclose(sum_list_test, scale).all())

        # check loaded train
        self.assertTrue((loaded_train_scale_expected == self.preprocessing.loaded_train_data).all())
        # check loaded map
        self.assertTrue((loaded_test_scale_expected == self.preprocessing.loaded_test_data).all())


#
# MMMTestnp
#
class UtilityTest(unittest.TestCase):

    def test_log_factorial(self):
        expected_1 = 363.73937555556347
        result_1 = nb.Utility.log_factorial(100)
        self.assertEqual(expected_1, result_1, 'test_log_factorial: Fail on test ln(100!)')

        expected_2 = 359.1342053695754
        result_2 = nb.Utility.log_factorial(99)
        self.assertEqual(expected_2, result_2, 'test_log_factorial: Fail on test ln(99!)')

        expected_3 = 51.60667556776438
        result_3 = nb.Utility.log_factorial(23)
        self.assertEqual(expected_3, result_3, 'test_log_factorial: Fail on test ln(23!)')

    def test_multinomial_and_posteriori_estimate(self):
        # The method using in NBText is a approximate estimation method.
        # Then this test should check the difference with actual calculation is smaller than acceptable epsilon
        eps = 0.0001

        # multinomial test
        word_prior_1 = np.asanyarray([0.1, 0.5, 0.2, 0.2])
        x_1 = np.random.multinomial(4, word_prior_1)
        expected_1 = scipy.stats.multinomial(n=4, p=word_prior_1).pmf(x_1)
        result_1 = nb.Utility.multinomial(x_1, word_prior_1)
        self.assertTrue(abs(expected_1 - result_1) < eps, 'test_multinomial: Fail on test 1')

        word_prior_2 = np.asanyarray([0.5, 0.1, 0.1, 0.3])
        x_2 = np.random.multinomial(4, word_prior_2)
        expected_2 = scipy.stats.multinomial(n=4, p=word_prior_2).pmf(x_2)
        result_2 = nb.Utility.multinomial(x_2, word_prior_2)
        self.assertTrue(abs(expected_2 - result_2) < eps, 'test_multinomial: Fail on test 2')

        word_prior_3 = np.asanyarray([0.2, 0.2, 0.2, 0.4])
        x_3 = np.random.multinomial(4, word_prior_3)
        expected_3 = scipy.stats.multinomial(n=4, p=word_prior_3).pmf(x_3)
        result_3 = nb.Utility.multinomial(x_3, word_prior_3)
        self.assertTrue(abs(expected_3 - result_3) < eps, 'test_multinomial: Fail on test 3')

        # posteriori_estimate test
        expected_1 = np.prod(np.power(word_prior_1, x_1))
        result_1 = nb.Utility.posteriori_estimate(x_1, word_prior_1)
        self.assertTrue(abs(expected_1 - result_1) < eps, 'test_posteriori_estimate: Fail on test 1')

        expected_2 = np.prod(np.power(word_prior_2, x_2))
        result_2 = nb.Utility.posteriori_estimate(x_2, word_prior_2)
        self.assertTrue(abs(expected_2 - result_2) < eps, 'test_posteriori_estimate: Fail on test 2')

        expected_3 = np.prod(np.power(word_prior_3, x_3))
        result_3 = nb.Utility.posteriori_estimate(x_3, word_prior_3)
        self.assertTrue(abs(expected_3 - result_3) < eps, 'test_posteriori_estimate: Fail on test 3')


class AgglomerativeTreeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.vector_test1 = nb.hierarchy_tree(sum_vector=np.full((1, 5),1)[0], element_id_list=[1])
        cls.vector_test2 = nb.hierarchy_tree(sum_vector=np.arange(5).reshape(1, 5)[0], element_id_list=[1])
        cls.data_group = [
            nb.hierarchy_tree(sum_vector=np.asarray([1,2,3,4]), element_id_list=[0], splitter_list=[]),
            nb.hierarchy_tree(sum_vector=np.asarray([1,2,3,4]), element_id_list=[1], splitter_list=[]),
            nb.hierarchy_tree(sum_vector=np.asarray([2,8,6,4]), element_id_list=[2], splitter_list=[]),
            nb.hierarchy_tree(sum_vector=np.asarray([2,8,6,4]), element_id_list=[3], splitter_list=[]),
            nb.hierarchy_tree(sum_vector=np.asarray([4,5,3,4]), element_id_list=[4], splitter_list=[]),
            nb.hierarchy_tree(sum_vector=np.asarray([4,5,3,4]), element_id_list=[5], splitter_list=[]),
            nb.hierarchy_tree(sum_vector=np.asarray([7,1,2,6]), element_id_list=[6], splitter_list=[]),
            nb.hierarchy_tree(sum_vector=np.asarray([7,1,2,6]), element_id_list=[7], splitter_list=[]),
            nb.hierarchy_tree(sum_vector=np.asarray([4,2,7,4]), element_id_list=[8], splitter_list=[]),
            nb.hierarchy_tree(sum_vector=np.asarray([4,2,7,4]), element_id_list=[9], splitter_list=[])
        ]
        cls.empty_data = nb.SslDataset()

    def test_distance_function(self):
        """
        vector1 = [1, 1, 1, 1, 1]
        vector2 = [0, 1, 2, 3, 4]

        * bin-to-bin:
        |vector1 - vector2| = [1, 0, 1, 2, 3]
        |vector1 + vector2| = [1, 2, 3, 4, 5]
        distance = 1/2 + 0/4 + 1/6 + 4/8 + 9/10 = 1.9 + 1/6

        * match:
        mass vector1 = [1, 2, 3, 4, 5]
        mass vector2 = [0, 1, 3, 6, 10]
        distance = 1 + 1 + 0 + 2 + 5 = 9

        """
        bin_bin = nb.AgglomerativeTree.bin_bin_distance(self.vector_test1, self.vector_test2)
        match = nb.AgglomerativeTree.match_distance(self.vector_test1, self.vector_test2)
        self.assertEqual(1.9 + 1/6, bin_bin, 'bin_to_bin distance mismatch')
        self.assertEqual(9, match, 'match distance mismatch')

    def test_build_hierarchy_tree(self):
        """
        data_group =[
                    [1,2,3,4] [1,2,3,4]
                    [2,8,6,4] [2,8,6,4]
                    [4,5,3,4] [4,5,3,4]
                    [7,1,2,6] [7,1,2,6]
                    [4,2,7,4] [4,2,7,4]]

        * bin_to_bin distance
        Agglomerative tree and merged distances
            Splitter id		  0   7   1   5	  2	  6   4	  8	  3
                              |   |   |	  |   |   |   |   |   |
                            0	1	2	3	4	5	8	9	6	7
                            |	|	|	|	|	|	|	|	|	|
                             \ / 	 \ /	 \ /	 \ /	 \ /
            Layer 0			  0		  0		  0		  0		  0
                              |		  |		  |		  |		  |
                              |		   \	 /		  |		  |
                              |			\   /		  |		  |
            Layer 1			  |		 1,(179487)	      |		  |
                              |			  |           |		  |
                              |			   \		 /		  |
                              |				\		/		  |
            Layer 2			  |			   1,534344172		  |
                              |				    |			  |
                               \			   /			  |
                                \			  /				  |
            Layer 3			      1,59772893				  |
                                       |       				  |
                                        \					 /
                                         \					/
                                          \	   			   /
            Layer 4			                  2,69241961

        * match distance
        Agglomerative tree and merged distances
            Splitter id		  0   7   1   5	  2	  6   4	  8	  3
                              |   |   |	  |   |   |   |   |   |
                            0	1	2	3	4	5	8	9	6	7
                            |	|	|	|	|	|	|	|	|	|
                             \ / 	 \ /	 \ /	 \ /	 \ /
            Layer 0			  0		  0		  0		  0		  0
                              |		  |		  |		  |		  |
                              |		  |		   \	 /		  |
                              |		  |			\   /		  |
            Layer 1			  |		  |	      	  5			  |
                              |		  |           |		      |
                              |		  |	   		   \		 /
                              |		  |    			\		/
            Layer 2			  |		  |		 		   6.5
                              |		  |		       		|
                              |		   \			   /
                              |		    \			  /
            Layer 3			  |			     13.(3)
                              |       		   |
                               \	     	  /
                                \			 /
                                 \	   	    /
            Layer 4			         22.5
        """
        # test bin-bin
        bin_to_bin_expected = nb.hierarchy_tree(sum_vector=np.asarray([36, 36, 42, 44]),
                                                element_id_list=[0, 1, 2, 3, 4, 5, 8, 9, 6, 7],
                                                splitter_list=[nb.splitter(id=0, order=0), nb.splitter(id=2, order=1),
                                                               nb.splitter(id=4, order=2), nb.splitter(id=3, order=5),
                                                               nb.splitter(id=6, order=4), nb.splitter(id=5, order=6),
                                                               nb.splitter(id=1, order=7), nb.splitter(id=8, order=3),
                                                               nb.splitter(id=7, order=8)])
        model = nb.AgglomerativeTree(self.empty_data, 'bin_bin_distance')
        bin_bin_data = copy.deepcopy(self.data_group)
        result = model.build_hierarchy_tree(bin_bin_data)
        self.assertTrue(np.array_equal(bin_to_bin_expected.sum_vector, result.sum_vector),
                        'Agglomerative tree bin_to_bin distance: sum_vector assertion fail')
        self.assertTrue(bin_to_bin_expected.element_id_list == result.element_id_list,
                        'Agglomerative tree bin_to_bin distance: element_id_list assertion fail')
        self.assertTrue((len(bin_to_bin_expected) == len(result)) and
                        all([x==y for x, y in zip(bin_to_bin_expected.splitter_list, result.splitter_list)]),
                        'Agglomerative tree bin_to_bin distance: splitter_list assertion fail')
        # test match
        march_expected = nb.hierarchy_tree(sum_vector=np.asarray([36, 36, 42, 44]),
                                           element_id_list=[0, 1, 2, 3, 4, 5, 8, 9, 6, 7],
                                           splitter_list=[nb.splitter(id=0, order=0), nb.splitter(id=2, order=1),
                                                          nb.splitter(id=4, order=2), nb.splitter(id=6, order=4),
                                                          nb.splitter(id=5, order=5), nb.splitter(id=8, order=3),
                                                          nb.splitter(id=7, order=6), nb.splitter(id=3, order=7),
                                                          nb.splitter(id=1, order=8)])

        model = nb.AgglomerativeTree(self.empty_data, 'match_distance')
        match_data = copy.deepcopy(self.data_group)
        result = model.build_hierarchy_tree(match_data)
        self.assertTrue(np.array_equal(march_expected.sum_vector, result.sum_vector),
                        'Agglomerative tree march distance: sum_vector assertion fail')
        self.assertTrue(march_expected.element_id_list == result.element_id_list,
                        'Agglomerative tree march distance: element_id_list assertion fail')
        self.assertTrue((len(march_expected) == len(result)) and
                        all([x == y for x, y in zip(march_expected.splitter_list, result.splitter_list)]),
                        'Agglomerative tree march distance: splitter_list assertion fail')


class DataTestGenerator(object):

    default_export_path = 'test_data/'

    def __init__(self, list_word_pr_list, list_prior_pr,
                 train_size=100, total_word_count = 100, test_size_per_class = 10):
        """
        This generates data for test multinomial model. The generated data is returned as instant var data_list.
        Note:
        - list_word_prior_list and size_list should have the same number of elements (number of class)
        - each word_prior_list should be summed to 1. If not, it follows numpy.random.multinomial rule
        - The index of each word_prior_list will be the class label
        - the size of axis 1 of list_word_prior_list is the feature number.
        - Test file has the same instances for each class, defined in  test_data_per_class

        :param list_word_pr_list: ndarray, each element is a list of word probabilities for for a class.
        :param list_prior_pr: ndarray, prior list for each class
        :param train_size: training size
        :param total_word_count: length of document, default is 100
        :param test_size_per_class: test data will create with same instant per class, default is 10
        """
        try:
            if len(list_word_pr_list) != len(list_prior_pr):
                raise(SelfException.DataInputMismatchLength,
                      'DataTestGenerator: list_word_prior_list must, list_prior_pr must have same size')
            if type(list_word_pr_list) is not np.ndarray or type(list_prior_pr) is not np.ndarray:
                raise(SelfException.DataTypeConstraint,
                      'DataTestGenerator: word_prior_list and size_list must be numpy.ndarray')

            # data properties
            self.list_word_pr_list = list_word_pr_list
            self.list_prior_pr = list_prior_pr
            self.train_size = train_size
            self.train_size_list = list_prior_pr * train_size
            self.total_word_count = total_word_count
            self.feature_number = len(list_word_pr_list[0]) # TODO: Check case when there is an empty array
            self.class_number = len(list_prior_pr)
            self.test_size_per_class = test_size_per_class
            self.map_file = ''
            self.train_file = ''
            self.test_file = ''

            self.train_data_list =[]
            self.test_data_list =[]

            for counter, (size, word_pr_list) in enumerate(zip(self.train_size_list, self.list_word_pr_list)):
                # generate train data
                new_data = np.random.multinomial(self.total_word_count, word_pr_list, int(size))
                new_data_label = np.full((int(size), 1), counter)
                self.train_data_list.append(np.concatenate((new_data, new_data_label), axis=1))
                # generate test data
                new_data = np.random.multinomial(self.total_word_count, word_pr_list, self.test_size_per_class)
                new_data_label = np.full((self.test_size_per_class, 1), counter)
                self.test_data_list.append(np.concatenate((new_data, new_data_label), axis=1))

        except SelfException.DataInputMismatchLength as e:
            e.recall_traceback(sys.exc_info())

        except SelfException.DataTypeConstraint as e:
            e.recall_traceback(sys.exc_info())

        except BaseException:
            print('Unknown error!')
            raise

    def csv_export(self, pre_dir, extend_mode=False):
        """
        export data_list to csv file
        :param pre_dir: prefix location for test data
        :param extend_mode: bool, set true if extend train and test files for test many to one
        :return:
        """
        self.map_file = pre_dir + self.default_export_path + 'map.csv'
        self.train_file = pre_dir + self.default_export_path + 'train.csv'
        self.test_file = pre_dir + self.default_export_path + 'test.csv'
        # map, train, test save
        if extend_mode:
            with open(self.train_file, 'a') as f:
                np.savetxt(f, np.vstack(self.train_data_list)[:], delimiter=',', fmt="%s")
            with open(self.test_file, 'a') as f:
                np.savetxt(f, np.vstack(self.test_data_list)[:], delimiter=',', fmt="%s")
        else:
            np.savetxt(self.map_file, np.arange(self.class_number).reshape(1, self.class_number)[:],
                       delimiter=',', fmt="%s")
            np.savetxt(self.train_file, np.vstack(self.train_data_list)[:], delimiter=',', fmt="%s")
            np.savetxt(self.test_file, np.vstack(self.test_data_list)[:], delimiter=',', fmt="%s")


class MultinomialAllLabeledTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up test data with 4 class

        |w| = 4
        sum(x_k) = 100
        test instances per class = 10

        P(y) = [0.2, 0.5, 0.15 0.15]
        P(w | y) = [
            [0.25, 0.25, 0.25, 0.25],
            [0.3, 0.4, 0.2, 0.1],
            [0.2, 0.2, 0.2, 0.4],
            [0.1, 0.3, 0.3, 0.3]
        ]

        This test creates random multinomial data then using model to train this data
        The expected value is the loss data smaller than an small epsilon.
        Further more, when the data size is increased, the accuracy should have the same behaviour
        """
        cls.test_generator = DataTestGenerator(
            np.vstack(
                [np.asarray([0.25, 0.25, 0.25, 0.25]),
                 np.asarray([0.3, 0.4, 0.2, 0.1]),
                 np.asarray([0.2, 0.2, 0.2, 0.4]),
                 np.asarray([0.1, 0.3, 0.3, 0.3])]),
            np.asarray([0.2, 0.5, 0.15, 0.15]),
            train_size=1000,
            total_word_count=100,
            test_size_per_class=10)
        cls.test_generator.csv_export('MMM/')
        list_file = [cls.test_generator.map_file, cls.test_generator.train_file, cls.test_generator.test_file]

        # Extract data
        data = nb.Dataset()
        # [ map, train, test ]
        data.load_from_csv(list_file)
        data_ssl = nb.SslDataset(data, 0.1)

        # [dataset]
        cls.model = nb.MultinomialAllLabeled(data_ssl)
        cls.model.train()
        cls.model.test()

    def test_argument_estimate(self):
        # show info
        print('Multinomial AllLabeled Test')
        print('acc', metrics.accuracy_score(self.model.data.test_y, self.model.predicted_label))
        print('prior pr')
        print(self.model.prior_pr)
        print('word pr')
        print(self.model.word_pr)

        # test prior pr
        epsilon = 0.01
        for counter, (x, y) in enumerate(zip(self.model.prior_pr, self.test_generator.list_prior_pr)):
            self.assertTrue(abs(x - y) < epsilon,
                            'test_argument_estimate: prior probability of class ' + str(counter) + ' mismatch.')
        # test word pr, comparing vector word pr
        for class_counter, (class_model, class_expected) in \
                enumerate(zip(self.model.word_pr, self.test_generator.list_word_pr_list)):
            diff = np.abs((class_model - class_expected)).sum()
            self.assertTrue(diff < epsilon * self.test_generator.class_number,
                            'test_argument_estimate: word probability of class ' + str(class_counter) + ' mismatch.')


class MultinomialEMTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up test data with 4 class, this temporarily has the same test method with all labeled

        |w| = 4
        sum(x_k) = 100
        test instances per class = 10

        P(y) = [0.2, 0.5, 0.15, 0.15]
        P(w | y) = [
            [0.25, 0.25, 0.25, 0.25],
            [0.3, 0.4, 0.2, 0.1],
            [0.2, 0.2, 0.2, 0.4],
            [0.1, 0.3, 0.3, 0.3]
        ]

        This test creates random multinomial data then using model to train this data
        The expected value is the loss data smaller than an small epsilon.
        Further more, when the data size is increased, the accuracy should have the same behaviour
        """
        cls.test_generator = DataTestGenerator(
            np.vstack(
                [np.asarray([0.25, 0.25, 0.25, 0.25]),
                 np.asarray([0.3, 0.4, 0.2, 0.1]),
                 np.asarray([0.2, 0.2, 0.2, 0.4]),
                 np.asarray([0.1, 0.3, 0.3, 0.3])]),
            np.asarray([0.2, 0.5, 0.15, 0.15]),
            train_size=1000,
            total_word_count=100,
            test_size_per_class=10)
        cls.test_generator.csv_export('MMM/')
        list_file = [cls.test_generator.map_file, cls.test_generator.train_file, cls.test_generator.test_file]

        # Extract data
        data = nb.Dataset()
        # [ map, train, test ]
        data.load_from_csv(list_file)
        data_ssl = nb.SslDataset(data, unlabeled_size=.4)

        # [dataset]
        cls.model = nb.MultinomialEM(data_ssl)
        cls.model.train()
        cls.model.test()

    def test_argument_estimate(self):
        # show info
        print('Multinomial EM Test')
        print('EM loops: ', self.model.EM_loop_count)
        print('acc', metrics.accuracy_score(self.model.data.test_y, self.model.predicted_label))
        print('prior pr')
        print(self.model.prior_pr)
        print('word pr')
        print(self.model.word_pr)

        # test prior pr
        epsilon = 0.01
        for counter, (x, y) in enumerate(zip(self.model.prior_pr, self.test_generator.list_prior_pr)):
            self.assertTrue(abs(x - y) < epsilon,
                            'test_argument_estimate: prior probability of class ' + str(counter) + ' mismatch.')
        # test word pr, comparing vector word pr
        for class_counter, (class_model, class_expected) in \
                enumerate(zip(self.model.word_pr, self.test_generator.list_word_pr_list)):
            diff = np.abs((class_model - class_expected)).sum()
            self.assertTrue(diff < epsilon * self.test_generator.class_number,
                            'test_argument_estimate: word probability of class ' + str(class_counter) + ' mismatch.')


class MultinomialManyToOneTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        This first test has the same test method with all labeled

        [Test case 1]
        Set up test data with 4 class
        |w| = 4
        sum(x_k) = 100
        test instances per class = 10

        P(y) = [0.2, 0.5, 0.15, 0.15] # instances per class follows this proportion
        P(w | y) = [
            [0.25, 0.25, 0.25, 0.25],
            [0.3, 0.4, 0.2, 0.1],
            [0.2, 0.2, 0.2, 0.4],
            [0.1, 0.3, 0.3, 0.3]
        ]

        This test creates random multinomial data then using model to train this data
        The expected value is the loss data smaller than an small epsilon.
        Further more, when the data size is increased, the accuracy should have the same behaviour

        [Test case 2]
        Set up the second test generate and merge it with the fist one. Then we have same label,
        with different word distributions.
        |w2| = 4 # this must be the same with the first 1
        sum(x_k) = 100
        test instances per class = 10

        P(y) = [0.2, 0.3, 0.3, 0.2]

        P(w | y) = [
            [0.1, 0.1, 0.2, 0.6],
            [0.2, 0.5, 0.1, 0.2],
            [0.1, 0.2, 0.3, 0.4],
            [0.6, 0.1, 0.1, 0.2]
        ]

        """
        cls.test_generator_1 = DataTestGenerator(
            np.vstack(
                [np.asarray([.25, .25, .25, .25]),
                 np.asarray([.3, .4, .2, .1]),
                 np.asarray([.2, .2, .2, .4]),
                 np.asarray([.1, .3, .3, .3])]),
            np.asarray([.2, .5, .15, .15]),
            train_size=1000,
            total_word_count=100,
            test_size_per_class=20)

        cls.test_generator_2 = DataTestGenerator(
            np.vstack(
                [np.asarray([.1, .1, .2, .6]),
                 np.asarray([.2, .5, .1, .2]),
                 np.asarray([.1, .2, .3, .4]),
                 np.asarray([.6, .1, .1, .2])]),
            np.asarray([.2, .3, .3, .2]),
            train_size=1000,
            total_word_count=100,
            test_size_per_class=20)

    def tes_equal_sampling(self):
        """
        test sum of all elements is 1
        :return:
        """
        test_size_1 = 10
        result_1 = nb.MultinomialManyToOne.equal_sampling(test_size_1)
        self.assertEqual(1, result_1.sum(), 'test_equal_sampling: Fail in test 1')

        test_size_2 = 100
        result_2 = nb.MultinomialManyToOne.equal_sampling(test_size_2)
        self.assertEqual(1, result_2.sum(), 'test_equal_sampling: Fail in test 2')

        test_size_3 = 1234
        result_3 = nb.MultinomialManyToOne.equal_sampling(test_size_3)
        self.assertEqual(1, result_3.sum(), 'test_equal_sampling: Fail in test 3')

    def tes_argument_estimate_one_one_component(self):
        """
        Simpple test with only one component per class.
        The expected result should be same as MultinomialEM
        :return:
        """
        self.test_generator_1.csv_export('MMM/')
        list_file = [self.test_generator_1.map_file, self.test_generator_1.train_file, self.test_generator_1.test_file]
        # Extract data
        data = nb.Dataset()
        # [ map, train, test ]
        data.load_from_csv(list_file)
        data_ssl = nb.SslDataset(data, unlabeled_size=.4)
        # [dataset, component_count_list, component_assignment_list=None]
        model = nb.MultinomialManyToOne(data_ssl, np.full((self.test_generator_1.class_number, 1), 1))
        model.train()
        model.test()

        # show info
        print('Multinomial ManyToOne one-one Test')
        print('EM loops: ', model.EM_loop_count)
        print('acc', metrics.accuracy_score(model.data.test_y, model.predicted_label))
        print('prior pr')
        print(model.prior_pr)
        print('word pr')
        print(model.word_pr)

        # test prior pr
        epsilon = 0.01
        for counter, (x, y) in enumerate(zip(model.prior_pr, self.test_generator_1.list_prior_pr)):
            self.assertTrue(abs(x - y) < epsilon,
                            'test_argument_estimate_one_one_component: prior probability of class '
                            + str(counter) + ' mismatch.')
        # test word pr, comparing vector word pr
        for class_counter, (class_model, class_expected) in \
                enumerate(zip(model.word_pr, self.test_generator_1.list_word_pr_list)):
            diff = np.abs((class_model - class_expected)).sum()
            self.assertTrue(diff < epsilon * self.test_generator_1.class_number,
                            'test_argument_estimate_one_one_component: word probability of class '
                            + str(class_counter) + ' mismatch.')

    def test_argument_estimate_many_one_component(self):
        """
        Test with equal number of components per class.
        The estimated value should not be larger than expected value more than epsilon
        :return:
        """
        self.test_generator_1.csv_export('MMM/')
        list_file = [self.test_generator_1.map_file, self.test_generator_1.train_file, self.test_generator_1.test_file]

        self.test_generator_2.csv_export('MMM/', extend_mode=True)

        # Extract data
        data = nb.Dataset()
        # [ map, train, test ]
        data.load_from_csv(list_file)
        data_ssl = nb.SslDataset(data, unlabeled_size=.4)
        # [dataset, component_count_list, component_assignment_list=None]
        # 2 component per class
        model = nb.MultinomialManyToOne(data_ssl, np.full((self.test_generator_1.class_number, 1), 2))
        model.train()
        model.test()

        # prior_pr or word_pr now are a list with components, each component has corresponding with
        # its data count and total data in all components.
        # Combine 2 generator for expected value
        expected_prior_pr = []
        # @@ OMG this has cost me a lot of time @@
        expected_word_pr = np.empty((self.test_generator_1.class_number * 2, self.test_generator_1.feature_number))
        train_size_1 = self.test_generator_1.train_size
        train_size_2 = self.test_generator_2.train_size
        total_train_data = train_size_1 + train_size_2
        for counter, (x, y) in enumerate(zip(self.test_generator_1.list_prior_pr, self.test_generator_2.list_prior_pr)):
            # we don't know which component of generator 1 or 2 first.
            # So we find the most match with model prior, same with word_pr
            sample_prior = model.prior_pr[2*counter]
            if (abs(train_size_1 * x / float(total_train_data) - sample_prior)
                    < abs(train_size_2 * y / float(total_train_data) - sample_prior)):
                expected_prior_pr.extend([train_size_1 * x / float(total_train_data),
                                          train_size_2 * y / float(total_train_data)])
            else:
                expected_prior_pr.extend([train_size_2 * y / float(total_train_data),
                                          train_size_1 * x / float(total_train_data)])
            # just in case we have same prior_pr
            if (np.abs(model.word_pr[2 * counter] - self.test_generator_1.list_word_pr_list[counter]).sum() <
                np.abs(model.word_pr[2*counter + 1] - self.test_generator_1.list_word_pr_list[counter]).sum()):
                expected_word_pr[2 * counter] = self.test_generator_1.list_word_pr_list[counter]
                expected_word_pr[2 * counter + 1] = self.test_generator_2.list_word_pr_list[counter]
            else:
                expected_word_pr[2 * counter] = self.test_generator_2.list_word_pr_list[counter]
                expected_word_pr[2 * counter + 1] = self.test_generator_1.list_word_pr_list[counter]

        # show info
        print('Multinomial ManyToOne many_one Test')
        print('EM loops: ', model.EM_loop_count)
        print('acc', metrics.accuracy_score(model.data.test_y, model.predicted_label))
        print('prior pr')
        print(model.prior_pr)
        print('expected prior pr')
        print(expected_prior_pr)
        print('word pr')
        print(model.word_pr)
        print('expected word pr')
        print(expected_word_pr)

        # test prior pr
        epsilon = 0.01
        for counter, (x, y) in enumerate(zip(model.prior_pr, expected_prior_pr)):
            self.assertTrue(abs(x - y) < epsilon,
                            'test_argument_estimate_many_one_component: prior probability of class '
                            + str(counter) + ' mismatch.')
        # test word pr, comparing vector word pr
        for class_counter, (class_model, class_expected) in enumerate(zip(model.word_pr, expected_word_pr)):
            diff = np.abs((class_model - class_expected)).sum()
            self.assertTrue(diff < epsilon * self.test_generator_1.class_number,
                            'test_argument_estimate_many_one_component: word probability of class '
                            + str(class_counter) + ' mismatch.')


#
# main
#
def suite(test_classes):
    suite_list = []
    loader = unittest.TestLoader()
    for test_class in test_classes:
        suite = loader.loadTestsFromTestCase(test_class)
        suite_list.append(suite)
    return suite_list


def main():
    # test list
    mmm_test = [UtilityTest, MultinomialAllLabeledTest, MultinomialEMTest,
                AgglomerativeTreeTest, MultinomialManyToOneTest]

    data_preprocessing_test = [Preprocessing20NewsTest]

    temp_test = [Preprocessing20NewsTest]

    # list of all desired tests
    require_test = 'mmm_test data_preprocessing_test'
    # require_test = 'temp_test'

    # print('Current supported test: [ MMM ]')
    # require_test = input("Test list: ")
    require_test = [x.lower() for x in require_test.split()]
    test_list = []
    if 'mmm_temp' in require_test:
        test_list.extend(temp_test)
    if 'mmm' in require_test:
        test_list.extend(mmm_test)
    if 'data_preprocessing' in require_test:
        test_list.extend(data_preprocessing_test)

    test_suite = unittest.TestSuite(suite(test_list))
    runner = unittest.TextTestRunner()
    runner.run(test_suite)


if __name__ == '__main__':
    main()