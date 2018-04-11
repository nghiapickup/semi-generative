# Data preprocessing for semi supervised code
# data format can be retrieved in readme
#
# This code is a mess stuff !!!
#
# @author: nghianh | Yamada lab

import os
import sys
import collections
import exceptionHandle as SelfException
import numpy as np
from sklearn import model_selection
import logging

logger = logging.getLogger(__name__)
SelfException.LogHandler(__name__)


class PreprocessingSample(object):
    __doc__ = 'Data preprocessing or Iris and Abalone data'

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


file_location_list = collections.namedtuple('file_location_list', 'vocabulary_file, map_input, '
                                                                  'train_input, train_label_input, '
                                                                  'test_input, test_label_input, '
                                                                  'map_output, train_output, test_output, data_info')

class Preprocessing20News(object):
    __doc__ = '20News data pre-processing.'

    reuters_stop_word_file = 'reuters_wos.txt'
    mi_word_rank_file = '20news-bydate/mi_word_rank.txt'
    required_parameter = 1

    def __init__(self, subfolder=''):
        """
        init common attribute
        :param subfolder: sub folder for output files
        """
        # TODO check sub-folder type is string

        self.file_list = file_location_list('20news-bydate/vocabulary.txt', '20news-bydate/data.map',
                                            '20news-bydate/train.data', '20news-bydate/train.label',
                                            '20news-bydate/test.data', '20news-bydate/test.label',
                                            '20news-bydate/final/' + subfolder + '/news.map.csv',
                                            '20news-bydate/final/' + subfolder + '/news.train.csv',
                                            '20news-bydate/final/' + subfolder + '/news.test.csv',
                                            '20news-bydate/final/' + subfolder + '/data_info.txt')
        self.loaded_map_data = np.empty(0)
        self.loaded_train_data = np.empty(0)
        self.loaded_test_data = np.empty(0)

    def data_csv_export(self, map_data, train_data, test_data):
        """
        extract data to files
        :param map_data:
        :param train_data:
        :param test_data:
        :return:
        """
        # the first exitst_ok shoule be False, this makes sure that all dir is empty before creating new data files
        os.makedirs(os.path.dirname(self.file_list.train_output), exist_ok=False)
        with open(self.file_list.train_output, 'w') as f:
            np.savetxt(f, train_data[:], fmt="%s", delimiter=',')

        os.makedirs(os.path.dirname(self.file_list.train_output), exist_ok=True)
        with open(self.file_list.test_output, 'w') as f:
            np.savetxt(f, test_data[:], fmt="%s", delimiter=',')

        os.makedirs(os.path.dirname(self.file_list.train_output), exist_ok=True)
        with open(self.file_list.map_output, 'w') as f:
            np.savetxt(f, np.mat(map_data)[0], fmt="%s", delimiter=',')

        os.makedirs(os.path.dirname(self.file_list.train_output), exist_ok=True)
        with open(self.file_list.data_info, 'w') as f:
            f.write('train number ' + str(np.shape(train_data)[0]) + '\n')
            f.write('test number ' + str(np.shape(test_data)[0]) + '\n')
            f.write('feature number ' + str(np.shape(train_data)[1] - 1) + '\n')

    def news_data_basic_process(self, scale_length=-1, extract_to_file=False):
        """
        Tokenize 20news data, only stemming, remove stop words and one time occurrence words.
        the data is used here is by-date and was splitted in train-test as .6-.4
        :param scale_length: default length of scaling for data, default is -1: no scale
        :param extract_to_file: bool, flag to raise extract processed data to files, default is true
        """
        # read data
        map_load = np.loadtxt(self.file_list.map_input, dtype='str', delimiter=',')
        train_load = np.loadtxt(self.file_list.train_input, dtype='int')
        train_label_load = np.loadtxt(self.file_list.train_label_input, dtype='str')
        test_load = np.loadtxt(self.file_list.test_input, dtype='int')
        test_label_load = np.loadtxt(self.file_list.test_label_input, dtype='str')
        vocabulary_load = np.loadtxt(self.file_list.vocabulary_file, dtype='str')
        stop_word_load = np.loadtxt(self.reuters_stop_word_file, dtype='str')

        # re-index class to number 0, 1, ..., c
        index_map = {}
        for i in range(len(map_load)):
            index_map[map_load[i]] = i
        for index, val in enumerate(train_label_load):
            train_label_load[index] = index_map.get(val)
        for index, val in enumerate(test_label_load):
            test_label_load[index] = index_map.get(val)

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

        # scaling data with fix length
        # a = (a.T * scale / a.sum(axis=1)).T
        # We omit the label a.T[:-1].T before calculating
        # * Should do this before remove feature. It makes sure that there is no zero vector
        if scale_length > 0:
            train_data.T[:-1] = train_data.T[:-1] * scale_length / train_data.T[:-1].sum(axis=0)
            test_data.T[:-1] = test_data.T[:-1] * scale_length / test_data.T[:-1].sum(axis=0)

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

        # extract to files
        if extract_to_file:
            self.data_csv_export(map_load, train_data, test_data)

        self.loaded_map_data = map_load
        self.loaded_train_data = train_data
        self.loaded_test_data = test_data

    def mutual_information_export(self):
        """
        Compute Average Multual Information of words and rank by word id in vocabulary file.
        This only calculates the MI based on training file.
        * NOTICE:
        - The file formats as each for id per line
        - The order is descending. Word has larger average MI will be printed first
        - The id using here is counted from 0 (difference with raw train, test file)
        - [CAUTION] When using this list, do not mess the id.
        :return:
        """

        data_load = np.loadtxt(self.file_list.train_input, dtype='int')
        data_label_load = np.loadtxt(self.file_list.train_label_input, dtype='int')
        vocabulary_count = len(np.loadtxt(self.file_list.vocabulary_file, dtype='str'))
        class_number = len(np.loadtxt(self.file_list.map_input,delimiter=','))
        train_number = len(data_label_load)

        word_conditional_class_pr = np.zeros((class_number, vocabulary_count))

        # Notice that word and data id in raw data file counting form 1
        for (data_id, word_id, word_count) in data_load:
            word_conditional_class_pr[data_label_load[data_id-1]-1][word_id-1] += 1
        class_pr = np.zeros(class_number)
        class_unique, class_count = np.unique(data_label_load, return_counts=True)
        class_unique[:] -= 1
        class_pr[class_unique] = class_count
        word_pr = word_conditional_class_pr.sum(axis=0)

        class_pr = np.divide(class_pr, train_number)
        word_pr = np.divide(word_pr, train_number)
        word_conditional_class_pr = np.divide(word_conditional_class_pr, train_number)
        word_mi_rank = np.zeros(vocabulary_count)

        for w in range(vocabulary_count):
            for c in range(class_number):
                # check if class does not have any instance
                # check if word does not have any instance, p(c,w) also = 0
                # or word occurs in any class
                # or there is no w in class c
                if class_pr[c] != 0 and word_pr[w] != 0 and word_pr[w] != 1 and word_conditional_class_pr[c, w] != 0:
                    word_mi_rank[w] += word_conditional_class_pr[c, w] * \
                                    np.log2(word_conditional_class_pr[c, w] / (class_pr[c] * word_pr[w]))
                    word_mi_rank[w] += (1 - word_conditional_class_pr[c, w]) * \
                                    np.log2((1 - word_conditional_class_pr[c, w]) / (class_pr[c] * (1 - word_pr[w])))

        # export to file
        with open(self.mi_word_rank_file, 'w') as f:
            np.savetxt(f, word_mi_rank.argsort()[::-1][:vocabulary_count], fmt="%s")

        # for test case
        return (class_pr, word_pr, word_conditional_class_pr, word_mi_rank)

    def news_data_mi_selection_process(self, selected_word_number=300, scale_length=-1, extract_to_file=False):
        """
        Tokenize 20news data, only stemming, and choosing top selected_word_number with highest mutual information score.
        the data is used here is by-date and was splitted in train-test as .6-.4
        :param selected_word_number: number of features selected
        :param scale_length: default length of scaling for data, default is -1: no scale
        :param extract_to_file: bool, flag to raise extract processed data to files, default is true
        """
        # read data
        map_load = np.loadtxt(self.file_list.map_input, dtype='str', delimiter=',')
        train_load = np.loadtxt(self.file_list.train_input, dtype='int')
        train_label_load = np.loadtxt(self.file_list.train_label_input, dtype='str')
        test_load = np.loadtxt(self.file_list.test_input, dtype='int')
        test_label_load = np.loadtxt(self.file_list.test_label_input, dtype='str')
        vocabulary_load = np.loadtxt(self.file_list.vocabulary_file, dtype='str')
        mi_rank_list_load = np.loadtxt(self.mi_word_rank_file, dtype='int')

        # re-index class to number 0, 1, ..., c
        index_map = {}
        for i in range(len(map_load)):
            index_map[map_load[i]] = i
        for index, val in enumerate(train_label_load):
            train_label_load[index] = index_map.get(val)
        for index, val in enumerate(test_label_load):
            test_label_load[index] = index_map.get(val)

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

        # scaling data with fix length
        # a = (a.T * scale / a.sum(axis=1)).T
        # We omit the label a.T[:-1].T before calculating
        # * Should do this before remove feature. It makes sure that there is no zero vector
        if scale_length > 0:
            train_data.T[:-1] = train_data.T[:-1] * scale_length / train_data.T[:-1].sum(axis=0)
            test_data.T[:-1] = test_data.T[:-1] * scale_length / test_data.T[:-1].sum(axis=0)

        # only pick data in mi rank list
        # TODO: exception when selected_word_number > vocabulary size
        pick_id = mi_rank_list_load[:selected_word_number]
        pick_id = np.append(pick_id, -1) # add label
        # noting here the order of word is now follow the MI rank list
        train_data = train_data.T[pick_id].T
        test_data = test_data.T[pick_id].T

        # extract to files
        if extract_to_file:
            self.data_csv_export(map_load, train_data, test_data)

        self.loaded_map_data = map_load
        self.loaded_train_data = train_data
        self.loaded_test_data = test_data


def main():
    try:
        cmd_list_test = ['1a_test',
                         'news_data_mi_selection_process 100 1 extract_to_file=True',
                         'news_data_mi_selection_process 200 -1 extract_to_file=True']

        cmd_export_mi_list = ['', 'mutual_information_export']

        # [EXP]
        # 1.a. scale and no scale, with vary features number
        cmd_1a_scale = ['1a_scale',
                        'news_data_mi_selection_process 100 1 extract_to_file=True',
                        'news_data_mi_selection_process 200 1 extract_to_file=True',
                        'news_data_mi_selection_process 400 1 extract_to_file=True',
                        'news_data_mi_selection_process 600 1 extract_to_file=True',
                        'news_data_mi_selection_process 1000 1 extract_to_file=True',
                        'news_data_mi_selection_process 5000 1 extract_to_file=True',
                        'news_data_mi_selection_process 7000 1 extract_to_file=True',
                        'news_data_mi_selection_process 10000 1 extract_to_file=True']
        cmd_1a_no_scale = ['1a_no_scale',
                           'news_data_mi_selection_process 100 -1 extract_to_file=True',
                           'news_data_mi_selection_process 200 -1 extract_to_file=True',
                           'news_data_mi_selection_process 400 -1 extract_to_file=True',
                           'news_data_mi_selection_process 600 -1 extract_to_file=True',
                           'news_data_mi_selection_process 1000 -1 extract_to_file=True',
                           'news_data_mi_selection_process 5000 -1 extract_to_file=True',
                           'news_data_mi_selection_process 7000 -1 extract_to_file=True',
                           'news_data_mi_selection_process 10000 -1 extract_to_file=True']

        # list of cmd, with the first element is sub-folder name. This will be the sub dir of default dir.
        # FIXME alter here
        cmd_list = cmd_1a_scale
        # only accept cmd called function from this list
        list_accepted_function = 'news_data_basic_process mutual_information_export ' \
                                 'news_data_mi_selection_process'.split()

        sub_folder = cmd_list[0]
        logger.info('Export data list: ' + sub_folder)

        for counter, cmd in enumerate(cmd_list[1:], start=1):
            print(cmd)
            logger.info(cmd)

            cmd = cmd.split()
            # conditions checking
            if len(cmd) < Preprocessing20News.required_parameter:
                raise SelfException.MismatchInputArgumentList('Preprocessing20News requires at least ' +
                                                              str(Preprocessing20News.required_parameter)+' arguments.')
            if cmd[0] not in list_accepted_function:
                raise SelfException.NonExitstingFunction('Preprocessing20News called function does not exist.')

            data = Preprocessing20News(subfolder=sub_folder + str(counter))
            function_called = getattr(data, cmd[0])

            if cmd[0] == 'news_data_basic_process':
                function_called(scale_length=int(cmd[1]), extract_to_file=bool(cmd[2]))
            elif cmd[0] == 'mutual_information_export':
                function_called()
            elif cmd[0] == 'news_data_mi_selection_process':
                function_called(selected_word_number=int(cmd[1]), scale_length=int(cmd[2]),extract_to_file=bool(cmd[3]))

        print('Done!')
        logger.info('Done!')

    except SelfException.NonExitstingFunction as e:
        e.recall_traceback(sys.exc_info())
        raise

    except BaseException:
        raise


if __name__ == '__main__':
    main()