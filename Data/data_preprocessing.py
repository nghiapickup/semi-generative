# Data preprocessing for semi supervised code
# data format can be retrieved in readme
#
# This code is a mess stuff !!!
#
# @author: nghianh | Yamada lab

import sys
import numpy as np
from sklearn import model_selection
import os


class Preprocessing(object):
    __doc__ = 'Data preprocessing'

    def __init__(self):
        # constant
        self.test_size = 0.3
        self.train_unlabeled_size = 0.9

        # Iris data
        # input
        self.iris_data_file = 'Iris/iris.data.txt'
        self.iris_map_file = 'Iris/iris.class.txt'
        # output
        self.iris_output_map = 'Iris/final/iris.map.csv'
        self.iris_output_train = ['Iris/final/iris.train.label.csv', 'Iris/final/iris.train.unlabel.csv']
        self.iris_output_test = 'Iris/final/iris.test.csv'

        # 20 news data
        # input
        self.news_vocabulary_file = '20news-bydate/vocabulary.txt'
        self.news_map_file = '20news-bydate/data.map'
        self.news_train_data = '20news-bydate/train.data'
        self.news_train_label = '20news-bydate/train.label'
        self.news_test_data = '20news-bydate/test.data'
        self.news_test_label = '20news-bydate/test.label'
        # output
        self.news_output_map = '20news-bydate/final/news.map.csv'
        self.news_output_train_data = ['20news-bydate/final/news.train.label.csv',
                                       '20news-bydate/final/news.train.unlabeled.csv']
        self.news_output_test = '20news-bydate/final/news.test.csv'

        # Abalone data
        # input
        self.abalone_data_file = 'abalone/abalone.data'
        self.abalone_map_file = 'abalone/abalone.map'
        # output
        self.abalone_output_map = 'abalone.map.csv'
        self.abalone_output_train = ['abalone.train.label.csv', 'abalone.train.unlabeled.csv']
        self.abalone_output_test = 'abalone.test.csv'

    def IrisData(self, args):
        # init splitting size
        if (len(args) > 1):
            self.test_size = float(args[1])
        if (len(args) > 2):
            self.train_unlabeled_size = float(args[2])

        # read data
        data_load = np.loadtxt(self.iris_data_file, dtype='str', delimiter=',')
        map_load = np.genfromtxt(self.iris_map_file, dtype='str', delimiter=',')

        # re-index class to number 0, 1, ..., c
        index_map = {}
        for i in range(len(map_load)):
            index_map[map_load[i]] = i

        for i, d in enumerate(data_load):
            d[-1] = index_map.get(d[-1])

        # split data into 3 parts, nearly same proportion
        # first slpit train and test, then from train split to label and unlabel
        sss1 = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=0)
        train_indices = [] #just in case
        for train_indices, test_indices in sss1.split(data_load, data_load.T[-1]):
            np.savetxt(self.iris_output_test, data_load[test_indices], fmt="%s", delimiter=',')  # test first
        #
        if self.train_unlabeled_size == 0:
            np.savetxt(self.iris_output_train[0],data_load[train_indices], fmt='%s', delimiter=',') # only train
        else:
            sss2 = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=self.train_unlabeled_size, random_state=0)
            for train_label_indices, train_unlabel_indices in sss2.split(data_load[train_indices],
                                                                         data_load[train_indices].T[-1]):
                np.savetxt(self.iris_output_train[0], data_load[train_indices][train_label_indices], fmt='%s',
                           delimiter=',')
                np.savetxt(self.iris_output_train[1], data_load[train_indices][train_unlabel_indices], fmt='%s',
                           delimiter=',')

        # map file generate
        np.savetxt(self.iris_output_map, np.mat(map_load)[0], fmt="%s", delimiter=',')

    def News20Data(self, args):
        """
        Basic pre-processing for News20 data using bag of words representation
        (words count vector)
        :param args: argument list
        """
        if (len(args) > 1):
            self.test_size = float(args[1])
        if (len(args) > 2):
            self.train_unlabeled_size = float(args[2])

        # read data
        map_load = np.loadtxt(self.news_map_file, dtype='str', delimiter=',')
        data_train_load = np.loadtxt(self.news_train_data, dtype='str')
        data_train_label_load = np.loadtxt(self.news_train_label, dtype='str')
        data_test_load = np.loadtxt(self.news_test_data, dtype='str')
        data_test_label_load = np.loadtxt(self.news_test_label, dtype='str')

        vocabulary_load = np.loadtxt(self.news_vocabulary_file, dtype='str')
        features_number = len(vocabulary_load)

        # re-index class to number 0, 1, ..., c
        index_map = {}
        for i in range(len(map_load)):
            index_map[map_load[i]] = i

        for i, d in enumerate(data_train_label_load):
            data_train_label_load[i] = index_map.get(d)

        for i, d in enumerate(data_test_label_load):
            data_test_label_load[i] = index_map.get(d)

        # train data generate
        train_data_number = len(data_train_label_load)
        train_data = np.mat(np.zeros((train_data_number, features_number + 1), dtype=np.uint8)) # data frame

        #vector bag of words
        # tup ~ (doc_id,word_id,count) ; doc_id and word_id numbering from 1
        for i, tup in enumerate(data_train_load):
            if len(tup) != 3:
                print('Train file: line ',i,' error')
            train_data[int(tup[0]) - 1, int(tup[1]) - 1] = tup[2]

        for i in range(train_data_number):
            train_data[i, -1] = data_train_label_load[i]

        ###
        #print(train_data)
        ###
        if self.train_unlabeled_size == 0:
            np.savetxt(self.news_output_train_data[0], train_data, fmt="%s", delimiter=',') # train all
        else:
            # split train data into 2 parts (label, unlabel), nearly same proportion
            sss1 = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=self.train_unlabeled_size, random_state=0)
            for label_indices, unlabel_indices in sss1.split(train_data, data_train_label_load):
                np.savetxt(self.news_output_train_data[0], train_data[label_indices], fmt="%s", delimiter=',')
                np.savetxt(self.news_output_train_data[1],
                           train_data[unlabel_indices], fmt="%s", delimiter=',')  # remove label

        # test data generate
        test_data_number = len(data_test_label_load)
        test_data = np.mat(np.zeros((test_data_number, features_number + 1), dtype=np.uint8))  # data frame

        # vector bag of words
        # tup ~ (doc_id,word_id,count) ; doc_id and word_id numbering from 1
        for i, tup in enumerate(data_test_load):
            if len(tup) != 3:
                print('Test file: line ', i, ' error')
            test_data[int(tup[0]) - 1, int(tup[1]) - 1] = tup[2]

        for i in range(test_data_number):
            test_data[i, -1] = data_test_label_load[i]

        ###
        #print('\n',test_data)
        ###

        np.savetxt(self.news_output_test, test_data[:], fmt="%s", delimiter=',')

        # map file generate
        np.savetxt(self.news_output_map, np.mat(map_load)[0], fmt="%s", delimiter=',')

    def AbaloneData(self, args, split_number=1):
        # Only extract data class from class 5 to 15
        # init splitting size
        if (len(args) > 1):
            self.test_size = float(args[1])
        if (len(args) > 2):
            self.train_unlabeled_size = float(args[2])

        # read data
        data_load = np.loadtxt(self.abalone_data_file, dtype='str', delimiter=',')
        map_load = np.genfromtxt(self.abalone_map_file, dtype='str', delimiter=',')

        # re-index class to number 0, 1, ..., c
        index_map = {}
        for i in range(len(map_load)):
            index_map[map_load[i]] = i

        # remove data out of range
        indices = [i for (i, d) in enumerate(data_load) if index_map.get(d[-1]) == None]
        data_load = np.delete(data_load, indices, axis=0)

        for i, d in enumerate(data_load):
            d[-1] = index_map.get(d[-1])

        # convert sex column to numeric
        u, indices = np.unique((data_load.T)[0], return_inverse  = True)
        (data_load.T)[0] = indices

        # split data into 3 parts, nearly same proportion
        # first slpit train and test, then from train split to label and unlabel
        sss1 = model_selection.StratifiedShuffleSplit(n_splits=split_number, test_size=self.test_size, random_state=0)
        train_indices = [] #just in case

        split_count = 0
        for train_indices, test_indices in sss1.split(data_load, data_load.T[-1]):
            partial_folder_name = 'abalone/' + str(split_count) + '/'
            os.makedirs(partial_folder_name , exist_ok=True)

            np.savetxt(partial_folder_name + self.abalone_output_test,
                       data_load[test_indices], fmt="%s", delimiter=',')  # test first
            #
            if self.train_unlabeled_size == 0:
                np.savetxt(partial_folder_name + self.abalone_output_train[0],data_load[train_indices], fmt='%s', delimiter=',') # only train
            else:
                sss2 = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=self.train_unlabeled_size, random_state=0)
                for train_label_indices, train_unlabel_indices in sss2.split(data_load[train_indices],
                                                                             data_load[train_indices].T[-1]):
                    np.savetxt(partial_folder_name + self.abalone_output_train[0], data_load[train_indices][train_label_indices], fmt='%s',
                               delimiter=',')
                    np.savetxt(partial_folder_name + self.abalone_output_train[1], data_load[train_indices][train_unlabel_indices], fmt='%s',
                               delimiter=',')

            # map file generate
            np.savetxt(partial_folder_name + self.abalone_output_map, np.mat(map_load)[0], fmt="%s", delimiter=',')

            split_count += 1


class Preprocessing2(object):
    __doc__ = 'New data pre-processing. This only splits data in train-test'

    def __init__(self, test_size=0.3):
        # common attribute
        self.reuters_stop_word = 'reuters_wos.txt'
        self.test_size = test_size

        # 20 news data
        self.info_log = '20news-bydate/final/info.txt'
        # # input
        # self.news_vocabulary_file = '20news-bydate/vocabulary.txt'
        # self.news_map_file = '20news-bydate/data.map'
        # self.news_train_data = '20news-bydate/train.data'
        # self.news_train_label = '20news-bydate/train.label'
        # self.news_test_data = '20news-bydate/test.data'
        # self.news_test_label = '20news-bydate/test.label'
        # # output
        # self.news_output_map = '20news-bydate/final/news.map.csv'
        # self.news_output_train = '20news-bydate/final/news.train.csv'
        # self.news_output_test = '20news-bydate/final/news.test.csv'

        #  DEMO
        # input
        self.news_vocabulary_file = '20news-bydate/demo/vocabulary.txt'
        self.news_map_file = '20news-bydate/demo/data.map'
        self.news_train_data = '20news-bydate/demo/train.data'
        self.news_train_label = '20news-bydate/demo/train.label'
        self.news_test_data = '20news-bydate/demo/test.data'
        self.news_test_label = '20news-bydate/demo/test.label'
        # output
        self.news_output_map = '20news-bydate/demo/final/news.map.csv'
        self.news_output_train = '20news-bydate/demo/final/news.train.csv'
        self.news_output_test = '20news-bydate/demo/final/news.test.csv'

    def data_extract(self, map_data, train_data, test_data):
        """
        extract data to files
        :param map_data:
        :param train_data:
        :param test_data:
        :return:
        """
        np.savetxt(self.news_output_train, train_data[:], fmt="%s", delimiter=',')
        np.savetxt(self.news_output_test, test_data[:], fmt="%s", delimiter=',')
        np.savetxt(self.news_output_map, np.mat(map_data)[0], fmt="%s", delimiter=',')

        with open(self.info_log, 'w') as f:
            f.write('train number ' + str(np.shape(train_data)[0]) + '\n')
            f.write('test number ' + str(np.shape(test_data)[0]) + '\n')
            f.write('feature number ' + str(np.shape(train_data)[1]) + '\n')

    def news_data_basic_process(self, scale_length=7, extract_to_file=True):
        """
        Tokenize 20news data, only stemming, remove stop words and one time occurrence words.
        the data is used here is by-date and was splitted in train-test as .6-.4
        length of scaling for all document default is 7 because max word count is ~7000
        :param scale_length: default length of scaling for data, default is 7
        :param extract_to_file: flag to raise extract processed data to files, default is true
        """
        # read data
        map_load = np.loadtxt(self.news_map_file, dtype='str', delimiter=',')
        train_load = np.loadtxt(self.news_train_data, dtype='int')
        train_label_load = np.loadtxt(self.news_train_label, dtype='str')
        test_load = np.loadtxt(self.news_test_data, dtype='int')
        test_label_load = np.loadtxt(self.news_test_label, dtype='str')
        vocabulary_load = np.loadtxt(self.news_vocabulary_file, dtype='str')
        stop_word_load = np.loadtxt(self.reuters_stop_word, dtype='str')

        # re-index class to number 0, 1, ..., c
        index_map = {}
        for i in range(len(map_load)):
            index_map[map_load[i]] = i
        for id, val in enumerate(train_label_load):
            train_label_load[id] = index_map.get(val)
        for id, val in enumerate(test_label_load):
            test_label_load[id] = index_map.get(val)

        # data farm
        temp_feature_number = len(vocabulary_load)
        train_number = len(train_label_load)
        train_data = np.zeros((train_number, temp_feature_number + 1))
        test_number = len(test_label_load)
        test_data = np.zeros((test_number, temp_feature_number + 1))

        # data generate
        # vector bag of words
        # tup ~ (doc_id,word_id,count) ; doc_id and word_id numbering from 1
        try:
            # train
            for i, tup in enumerate(train_load):
                if len(tup) != 3:
                    raise IOError('Train file: line ', i, ' error')
                train_data[int(tup[0]) - 1, int(tup[1]) - 1] = tup[2]
            # label assign
            train_data[:, -1] = train_label_load[:]

            # test
            for i, tup in enumerate(test_load):
                if len(tup) != 3:
                    raise IOError('Test file: line ', i, ' error')
                test_data[int(tup[0]) - 1, int(tup[1]) - 1] = tup[2]
            test_data[:, -1] = test_label_load[:]
        except BaseException:
            raise

        # remove stop word & one time occurrence word
        # do this after transfer into mat then we no need to re-index words id
        remove_id = [id for id, val in enumerate(vocabulary_load) if val in stop_word_load]
        temp_sum_word = np.sum(train_data.T[: -1].T, axis=0)
        # just in case the one time occurrence word also is stop word
        temp_sum_word = [id for id, val in enumerate(temp_sum_word)
                         if val < 2 and id not in remove_id]
        remove_id = np.append(remove_id, temp_sum_word)
        train_data = np.delete(train_data, remove_id, axis=1)
        test_data = np.delete(test_data, remove_id, axis=1)

        # check count size for scaling estimate
        # x = [id for id, val in enumerate(np.sum(train_data.T[:-1].T, axis=1)) if val == 0]

        # scaling data with fix length
        train_data = (train_data.T / train_data.sum(axis=1)).T
        test_data = (test_data.T * scale_length / test_data.sum(axis=1)).T

        # extract to files
        if extract_to_file:
            self.data_extract(map_load, train_data, test_data)

def main():
    # try:
    #     terminal = sys.argv
    #     data = Preprocessing()
    #
    #     # command list
    #     terminal_command = {
    #         "iris": data.IrisData,
    #         "20news": data.News20Data,
    #         "abalone": data.AbaloneData
    #     }
    #
    #     if len(terminal) > 1:
    #         # terminal extract
    #         func = terminal_command.get(terminal[1], 'Data set is not found!')
    #         if not isinstance(func, str):
    #             func((terminal[1:]))
    #             print('Done!')
    #         else:
    #             print(func)
    # except:
    #     e = sys.exc_info()
    #     print(e[0], e[1])

    # debug
    data = Preprocessing2()
    data.news_data_basic_process(extract_to_file=False)

    print('Done!')

if __name__ == '__main__':
    main()