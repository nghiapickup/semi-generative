# Data script 20news data set
# @author: nghianh | Yamada lab

"""
This script uses to re-split 20news data from 20news-bydate dataset
"""
import numpy as np
import collections

file_dir_list = collections.namedtuple('file_location_list', 'vocabulary_file, map_input, '
                                                                  'train_input, train_label_input, '
                                                                  'test_input, test_label_input, '
                                                                  'data_info')

# Merge all data (without train-test split)
# Each class is sorted by date
merge_origin_file_dir = file_dir_list(
    vocabulary_file='20news-bydate/origin/merge_origin/vocabulary.txt',
    map_input='20news-bydate/origin/merge_origin/data.map',
    train_input='20news-bydate/origin/merge_origin/train.data',
    train_label_input='20news-bydate/origin/merge_origin/train.label',
    test_input='',
    test_label_input='',
    data_info='20news-bydate/origin/merge_origin/info.txt')

# Sorted by data and split with 60-40 train-test scale
bydate_origin_file_dir = file_dir_list(
    vocabulary_file='20news-bydate/origin/bydate_origin/vocabulary.txt',
    map_input='20news-bydate/origin/bydate_origin/data.map',
    train_input='20news-bydate/origin/bydate_origin/train.data',
    train_label_input='20news-bydate/origin/bydate_origin/train.label',
    test_input='20news-bydate/origin/bydate_origin/test.data',
    test_label_input='20news-bydate/origin/bydate_origin/test.label',
    data_info='20news-bydate/origin/bydate_origin/info.txt')

# Sorted by data and split with equal number of instances per class
equal_class_test_file_dir = file_dir_list(
    vocabulary_file='20news-bydate/origin/equal_class_test_data/vocabulary.txt',
    map_input='20news-bydate/origin/equal_class_test_data/data.map',
    train_input='20news-bydate/origin/equal_class_test_data/train.data',
    train_label_input='20news-bydate/origin/equal_class_test_data/train.label',
    test_input='20news-bydate/origin/equal_class_test_data/test.data',
    test_label_input='20news-bydate/origin/equal_class_test_data/test.label',
    data_info='20news-bydate/origin/equal_class_test_data/info.txt')


# Reminding that the counting principal of class and document id must be keep as same as origin data.
def merge_origin_data():
    """
    Merge train-test in 20news-bydate into 1 data file.
    The date order (class instance is continuously arrange with date order) is reserved.

    First, test and train data are read separately.
    Then, loop through each class and append train-test data into merge data (train first)
    :return:
    """
    train_data_load = np.loadtxt(bydate_origin_file_dir.train_input, dtype='int')
    train_label_data_load = np.loadtxt(bydate_origin_file_dir.train_label_input, dtype='int')
    test_data_load = np.loadtxt(bydate_origin_file_dir.test_input, dtype='int')
    test_label_data_load = np.loadtxt(bydate_origin_file_dir.test_label_input, dtype='int')
    vocabulary_load = np.loadtxt(bydate_origin_file_dir.vocabulary_file, dtype='str')
    map_load = np.loadtxt(bydate_origin_file_dir.map_input, dtype='int', delimiter=',')

    train_number = len(train_label_data_load)
    test_number = len(test_label_data_load)
    vocabulary_size = len(vocabulary_load)
    class_number = len(map_load)

    train_data = np.zeros((train_number, vocabulary_size))
    test_data = np.zeros((test_number, vocabulary_size))
    #read data
    for word_pack in train_data_load:
        [data_id, word_id, word_count] = word_pack
        train_data[data_id-1, word_id-1] = word_count
    for word_pack in test_data_load:
        [data_id, word_id, word_count] = word_pack
        test_data[data_id-1, word_id-1] = word_count

    #export data
    # this is quite dumb method. But it makes sure that we do not miss if data not expected
    # data reading processes through class with train data first, test data later.]
    data_counter = 0
    with open(merge_origin_file_dir.train_input, 'w') as f_train,\
            open(merge_origin_file_dir.train_label_input, 'w') as f_test:
                for label in range(1, class_number + 1):
                    # train data
                    for data_id, data_label in enumerate(train_label_data_load):
                        if data_label == label:
                            data_counter += 1
                            f_test.writelines(str(label) + '\n')
                            for word_id in range(vocabulary_size):
                                if train_data[data_id, word_id] > 0:
                                    f_train.writelines(str(data_counter) + ' ' + str(word_id + 1) + ' ' +
                                                       str(int(train_data[data_id, word_id])) + '\n')
                    # test data
                    for data_id, data_label in enumerate(test_label_data_load):
                        if data_label == label:
                            data_counter += 1
                            f_test.writelines(str(label) + '\n')
                            for word_id in range(vocabulary_size):
                                if test_data[data_id, word_id] > 0:
                                    f_train.writelines(str(data_counter) + ' ' + str(word_id + 1) + ' ' +
                                                       str(int(test_data[data_id, word_id])) + '\n')

    return data_counter # for test


def equal_class_test_data_generator(test_instance_per_class=100):
    """
    split data from merge_origin_data with test data has equal number of instances per class
    :param test_instance_per_class: number, default=100, number of instances for each class in test data
    :return:
    """
    # TODO this function must be guagranteed that merge_origin data is exist
    # TODO test_instance_per_class must be possive

    data_load = np.loadtxt(merge_origin_file_dir.train_input, dtype='int')
    label_data_load = np.loadtxt(merge_origin_file_dir.train_label_input, dtype='int')
    vocabulary_load = np.loadtxt(merge_origin_file_dir.vocabulary_file, dtype='str')
    map_load = np.loadtxt(merge_origin_file_dir.map_input, dtype='int', delimiter=',')

    data_number = len(label_data_load)
    vocabulary_size = len(vocabulary_load)
    class_number = len(map_load)

    data = np.zeros((data_number, vocabulary_size))
    # read data
    for word_pack in data_load:
        [data_id, word_id, word_count] = word_pack
        data[data_id - 1, word_id - 1] = word_count

    # counting table
    unique_list, counter = np.unique(label_data_load, return_counts=True)
    count = np.zeros(class_number, dtype=int)
    # in case there is a class without instance
    for i in range(len(unique_list)):
        count[unique_list[i] - 1] = counter[i]

    counting_table = np.zeros(class_number + 1, dtype=int)
    for i in range(1, class_number + 1):
        counting_table[i] = counting_table[i-1] + count[i-1]

    # because that merge_origin data is sorted by class by day and arrange continuous by class
    # we simple export data by class
    # Notice the difference of indexing (from 1 or from 0)!
    train_export_count = 0
    test_export_count = 0
    with open(equal_class_test_file_dir.train_input, 'w') as f_train, \
            open(equal_class_test_file_dir.train_label_input, 'w') as f_train_label, \
            open(equal_class_test_file_dir.test_input, 'w') as f_test, \
            open(equal_class_test_file_dir.test_label_input, 'w') as f_test_label:
                for label in range(class_number):
                    # only work with class has intance
                    if count[label] > 0:
                        # compute how many instance left in train data
                        train_instance_number = count[label] - test_instance_per_class
                        # export train first
                        for data_id in range(counting_table[label], counting_table[label] + train_instance_number):
                            train_export_count += 1
                            f_train_label.writelines(str(label + 1) + '\n') # label index from 1
                            for word_id in range(vocabulary_size):
                                if data[data_id, word_id] > 0:
                                    f_train.writelines(str(train_export_count) + ' ' + str(word_id + 1) + ' ' +
                                                       str(int(data[data_id, word_id])) + '\n')
                        # export test
                        for data_id in range(counting_table[label] + train_instance_number, counting_table[label+1]):
                            test_export_count += 1
                            f_test_label.writelines(str(label + 1) + '\n') # label index from 1
                            for word_id in range(vocabulary_size):
                                if data[data_id, word_id] > 0:
                                    f_test.writelines(str(test_export_count) + ' ' + str(word_id + 1) + ' ' +
                                                       str(int(data[data_id, word_id])) + '\n')

def main():
    # merge_origin_data()
    equal_class_test_data_generator(test_instance_per_class=250)

    print('Done!')

if __name__ == '__main__':
    main()