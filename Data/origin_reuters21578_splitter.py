"""
Data script Reuters-21578 data set

The main part of SGML parser is adapted from:
--------------------------------------------
Author: QuantStart Team
Date: January 9th, 2015
Availability: https://www.quantstart.com/articles/Supervised-Learning-for-Document-Classification-with-Scikit-Learn
--------------------------------------------
"""

import html
import re
import os
from html.parser import HTMLParser
import collections
import logging

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

import exceptionHandle as selfException

logger = logging.getLogger('data_preprocessing')
selfException.LogHandler('data_preprocessing')

default_path = 'reuters21578/'
data_file = 'reuters.data'
label_file = 'reuters.label'
vocabulary_file = 'vocabulary.txt'
info_file = 'reuters21578_data_info.txt'
export_file_location_list = collections.namedtuple('file_location_list', 'root_dir, '
                                                                  'map_output, train_output, test_output, data_info')

class ReutersParser(HTMLParser):
    """
    ReutersParser subclasses HTMLParser and is used to open the SGML
    files associated with the Reuters-21578 categorised test collection.

    The parser is a generator and will yield a single document at a time.
    Since the data will be chunked on parsing, it is necessary to keep
    some internal state of when tags have been "entered" and "exited".
    Hence the in_body, in_topics and in_topic_d boolean members.
    """

    def __init__(self, encoding='latin-1'):
        """
        Initialise the superclass (HTMLParser) and reset the parser.
        Sets the encoding of the SGML files by default to latin-1.
        """
        html.parser.HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding

    def _reset(self):
        """
        This is called only on initialisation of the parser class
        and when a new topic-body tuple has been generated. It
        resets all off the state so that a new tuple can be subsequently
        generated.
        """
        self.in_body = False
        self.in_topics = False
        self.in_topic_d = False
        self.body = ""
        self.topics = []
        self.topic_d = ""

    def parse(self, fd):
        """
        parse accepts a file descriptor and loads the data in chunks
        in order to minimise memory usage. It then yields new documents
        as they are parsed.
        """
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_starttag(self, tag, attrs):
        """
        This method is used to determine what to do when the parser
        comes across a particular tag of type "tag". In this instance
        we simply set the internal state booleans to True if that particular
        tag has been found.
        """
        if tag == "reuters":
            pass
        elif tag == "body":
            self.in_body = True
        elif tag == "topics":
            self.in_topics = True
        elif tag == "d":
            self.in_topic_d = True

    def handle_endtag(self, tag):
        """
        This method is used to determine what to do when the parser
        finishes with a particular tag of type "tag".

        If the tag is a  tag, then we remove all
        white-space with a regular expression and then append the
        topic-body tuple.

        If the tag is a  or  tag then we simply set
        the internal state to False for these booleans, respectively.

        If the tag is a  tag (found within a  tag), then we
        append the particular topic to the "topics" list and
        finally reset it.
        """
        if tag == "reuters":
            self.body = re.sub(r'\s+', r' ', self.body)
            self.docs.append((self.topics, self.body))
            self._reset()
        elif tag == "body":
            self.in_body = False
        elif tag == "topics":
            self.in_topics = False
        elif tag == "d":
            self.in_topic_d = False
            self.topics.append(self.topic_d)
            self.topic_d = ""

    def handle_data(self, data):
        """
        The data is simply appended to the appropriate member state
        for that particular tag, up until the end closing tag appears.
        """
        if self.in_body:
            self.body += data
        elif self.in_topic_d:
            self.topic_d += data


class reuters21578(object):
    @staticmethod
    def obtain_topic_tags():
        """
        Open the topic list file and import all of the topic names
        taking care to strip the trailing "\n" from each word.
        """
        topics = open(
            default_path + "origin/all-topics-strings.lc.txt", "r"
        ).readlines()
        topics = [t.strip() for t in topics]
        return topics

    @staticmethod
    def filter_doc_list_through_topics(topics, docs):
        """
        Reads all of the documents and creates a new list of two-tuples
        that contain a single feature entry and the body text, instead of
        a list of topics. It removes all geographic features and only
        retains those documents which have at least one non-geographic
        topic.
        """
        ref_docs = []
        for d in docs:
            if d[0] == [] or d[0] == "":
                continue
            for t in d[0]:
                if t in topics:
                    d_tup = (t, d[1])
                    ref_docs.append(d_tup)
                    break
        return ref_docs

    @staticmethod
    def create_word_count_data(docs, basic_info_extract=False):
        """
        Creates a document corpus list (by stripping out the
        class labels), then applies the word count transform to this
        list.

        Label will be map to counter number

        The function returns both the class label vector (y) and
        the corpus token/feature matrix (X).
        """
        # Create the training data class labels
        y = np.asarray([d[0] for d in docs])

        # Create the document corpus list
        corpus = [d[1] for d in docs]

        # Create the word count vectoriser and transform the corpus
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)
        X = X.toarray()
        if basic_info_extract:
            reuters21578.extract_basic_info(X,y)

        # map label y to number set
        _, y_inverse = np.unique(y, return_inverse=True)
        # reshape y follows the matrix form
        y = y_inverse.reshape(len(y_inverse) ,1)
        return X, y

    @staticmethod
    def extract_basic_info(X,y):
        """
        Extract basic info
        :param X:
        :param y:
        :return:
        """
        y_unique, y_inverse, y_count = np.unique(y, return_inverse=True, return_counts=True)
        y_count_sort = np.argsort(y_count)

        with open(default_path+info_file, 'w') as f:
            f.write('#data: ' + str(X.shape[0]) + '\n')
            f.write('vocabulary size: ' + str(X.shape[1]) + '\n')
            f.write('Label counter' + '\n')
            f.write('label id - label name - counter' + '\n')
            for i in range(len(y_unique)):
                f.write(str(y_count_sort[i]) + '    ' + str(y_unique[y_count_sort[i]])
                        + '   ' + str(y_count[y_count_sort[i]]) + '\n')

    @staticmethod
    def reuters21578Extract(basic_info_extract=False):
        # Create the list of Reuters data and create the parser
        files = [default_path + "origin/reut2-%03d.sgm" % r for r in range(0, 22)]
        parser = ReutersParser()

        # Parse the document and force all generated docs into
        # a list so that it can be printed out to the console
        docs = []
        for fn in files:
            for d in parser.parse(open(fn, 'rb')):
                docs.append(d)

        # Obtain the topic tags and filter docs through it
        topics = reuters21578.obtain_topic_tags()
        ref_docs = reuters21578.filter_doc_list_through_topics(topics, docs)

        # Vectorise and word count transform the corpus
        return reuters21578.create_word_count_data(ref_docs, basic_info_extract)


class data_preprocessing(object):
    """
    preprocessing data
    """
    def __init__(self, X, y, test_size=.3, root_folder=''):
        """
        Initialize basic data
        :param X:
        :param y:
        :param test_size:
        :param subfolder:
        """
        self.file_list = export_file_location_list('reuters21578/final/' + root_folder,
                                                   '/news.map.csv', '/news.train.csv',
                                                   '/news.test.csv', '/data_info.txt')

        # extract data basic info
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=test_size, random_state=0)
        u = np.unique(y)
        self.class_number = len(u)
        self.vocabulary_number = X.shape[1]
        self.mi_rank_list = np.empty(self.vocabulary_number)

    def data_csv_export(self, map_data, train_data, test_data, subfolder=''):
        """
        extract data to files
        :param map_data:
        :param train_data:
        :param test_data:
        :return:
        """
        # the first exitst_ok shoule be False, this makes sure that all dir is empty before creating new data files
        train_file = self.file_list.root_dir + subfolder + self.file_list.train_output
        os.makedirs(os.path.dirname(train_file), exist_ok=True)
        with open(train_file, 'w') as f:
            np.savetxt(f, train_data[:], fmt="%s", delimiter=',')

        test_file = self.file_list.root_dir + subfolder + self.file_list.test_output
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        with open(test_file, 'w') as f:
            np.savetxt(f, test_data[:], fmt="%s", delimiter=',')

        map_file = self.file_list.root_dir + subfolder + self.file_list.map_output
        os.makedirs(os.path.dirname(map_file), exist_ok=True)
        with open(map_file, 'w') as f:
            np.savetxt(f, np.mat(map_data)[0], fmt="%s", delimiter=',')

        info_file = self.file_list.root_dir + subfolder + self.file_list.data_info
        os.makedirs(os.path.dirname(info_file), exist_ok=True)
        with open(info_file, 'w') as f:
            f.write('train number ' + str(np.shape(train_data)[0]) + '\n')
            f.write('test number ' + str(np.shape(test_data)[0]) + '\n')
            f.write('feature number ' + str(np.shape(train_data)[1] - 1) + '\n')

    def mutual_information_extract(self):
        """
        Compute Average Multual Information of words and extract train data with selected_word_number to file.

        * NOTICE:
        - The MI scrore only takes on train data

        :return:
        """
        logger.info('mutual_information_extract')
        vocabulary_occurrences_by_class_pr = []
        occurrences_count = []

        # loop on each vocabulary
        for vocabulary_intances in self.X_train.T:
            # get unique occurrences
            unique_occurrences, inverse_occurrences = np.unique(vocabulary_intances, return_inverse=True)
            # (*) No need to check occurrence '0', it counted already
            # add how many different occurrences of this vocabulary_id, use later for MI calc
            occurrences_count.append(len(unique_occurrences))

            # count occurences for each class
            occurrences_count_on_vocabulary = np.zeros((len(unique_occurrences), self.class_number))
            for occurrence_instance_id in range(len(vocabulary_intances)):
                # instance_indices and inverse_occurrences have same index
                # Notice that word and data id in raw data file counting form 1
                occurrences_count_on_vocabulary[inverse_occurrences[occurrence_instance_id],
                                                self.y_train[occurrence_instance_id]] += 1
            vocabulary_occurrences_by_class_pr.append(occurrences_count_on_vocabulary)

        vocabulary_occurrences_by_class_pr = np.vstack(vocabulary_occurrences_by_class_pr).T
        all_occurrences_number = vocabulary_occurrences_by_class_pr.sum()  # or class_number * vocabulary_number
        occurrences_pr = np.sum(vocabulary_occurrences_by_class_pr, axis=0)
        classes_pr = np.sum(vocabulary_occurrences_by_class_pr, axis=1)

        vocabulary_occurrences_by_class_pr = np.divide(vocabulary_occurrences_by_class_pr, all_occurrences_number)
        occurrences_pr = np.divide(occurrences_pr, all_occurrences_number)
        classes_pr = np.divide(classes_pr, all_occurrences_number)

        vocabulary_mi_rank = np.zeros(self.vocabulary_number)

        occurrence_count_id = 0
        for vocabulary_id in range(self.vocabulary_number):
            for occurrence_id in range(occurrence_count_id, occurrence_count_id + occurrences_count[vocabulary_id]):
                for class_id in range(self.class_number):
                    pr_class_occurrence = vocabulary_occurrences_by_class_pr[class_id, occurrence_id]
                    # check if class does not have this occurrence
                    # (both conditions below are to make sure the consistent
                    # check if occurrence does not have any instance
                    # check if class does not have any instance (no document assigns to this class)
                    if pr_class_occurrence != 0 and occurrences_pr[occurrence_id] != 0 and classes_pr[class_id] != 0:
                        vocabulary_mi_rank[vocabulary_id] += pr_class_occurrence * np.log2(
                            pr_class_occurrence / (classes_pr[class_id] * occurrences_pr[occurrence_id]))
            occurrence_count_id += occurrences_count[vocabulary_id]

        # save mi rank
        self.mi_rank_list = np.argsort(vocabulary_mi_rank)

        # for test case
        return classes_pr, occurrences_pr, vocabulary_occurrences_by_class_pr, self.mi_rank_list

    def data_mi_selection_export(self, selected_word_number=300, binary_test_class=-1, scale_length=-1,
                                 extract_to_file=False, subfolder=''):
        """
        Because the data is massive to save to file and re-read to handle.
        This function combines all necessary taks and exports the last processed data:
        """
        logger.info('data_mi_selection_export')
        # TODO Check Mi rank list existence

        # only pick data in mi rank list
        # TODO: exception when selected_word_number > vocabulary size
        pick_id = self.mi_rank_list[-selected_word_number:]
        # noting here the order of word is now follow the MI rank list
        train_data = self.X_train.T[pick_id].T
        test_data = self.X_test.T[pick_id].T
        train_label = self.y_train
        test_label = self.y_test

        # remove all zero vector
        train_zero_vector_list = [counter for counter, value in enumerate(train_data.T.sum(axis=0)) if value == 0]
        test_zero_vector_list = [counter for counter, value in enumerate(test_data.T.sum(axis=0)) if value == 0]
        train_data = np.delete(train_data, train_zero_vector_list, axis=0)
        train_label = np.delete(train_label, train_zero_vector_list, axis=0)
        test_data = np.delete(test_data, test_zero_vector_list, axis=0)
        test_label = np.delete(test_label, test_zero_vector_list, axis=0)

        # scaling data with fix length,scale must must be done after pick data from MI rank list
        # a = (a.T * scale / a.sum(axis=1)).T
        if scale_length > 0:
            train_data.T[:] = train_data.T[:] * scale_length / train_data.T.sum(axis=0)
            test_data.T[:] = test_data.T[:] * scale_length / test_data.T.sum(axis=0)

        # Transform binary classification base on binary_test_class
        if binary_test_class > -1:
            for counter, label in enumerate(train_label):
                train_label[counter] = 0 if train_label[counter] == binary_test_class else 1
            for counter, label in enumerate(test_label):
                test_label[counter] = 0 if test_label[counter] == binary_test_class else 1

        # Merge instance and label
        train = np.insert(train_data, train_data.shape[1], train_label.T, axis=1)
        test = np.insert(test_data, test_data.shape[1], test_label.T, axis=1)
        if binary_test_class > -1:
            map = np.arange(2)
        else:
            map = np.arange(self.class_number)

        # extract to files
        if extract_to_file:
            self.data_csv_export(map, train, test, subfolder=subfolder)

def main():
    # Get basic info
    X, y = reuters21578.reuters21578Extract(basic_info_extract=True)
    """
    id  class   # instances
    24  grain       537
    14  crude       543
    44  money-fx    682
    0   acq         2423
    17  earn        3972
    """
    logger.info('START REUTERS21578 DATA PREPROCESSING')
    logger.info('START preprocessing1: test_size=0.3')
    preprocessing1 = data_preprocessing(X, y, test_size=0.3,
                                        root_folder='2a_reuters_test_scale_3/')
    preprocessing1.mutual_information_extract()

    #
    # Test binary classification transform
    #
    # grain
    word_list = [400, 600, 800]
    for i in word_list:
        logger.info('preprocessing1 scale_length=10000, binary_test_class=24, selected_word_number=' + str(i))
        preprocessing1.data_mi_selection_export(selected_word_number=i, binary_test_class=24, scale_length=10000,
                                               extract_to_file=True, subfolder='grain'+str(i))

        # crude
        logger.info('preprocessing1 scale_length=10000, binary_test_class=14, selected_word_number=' + str(i))
        preprocessing1.data_mi_selection_export(selected_word_number=i, binary_test_class=14, scale_length=10000,
                                                extract_to_file=True, subfolder='crude'+str(i))

        # money-fx
        logger.info('preprocessing1 scale_length=10000, binary_test_class=44, selected_word_number=' + str(i))
        preprocessing1.data_mi_selection_export(selected_word_number=i, binary_test_class=44, scale_length=10000,
                                                extract_to_file=True, subfolder='money-fx'+str(i))

        # acq
        logger.info('preprocessing1 scale_length=10000, binary_test_class=0, selected_word_number=' + str(i))
        preprocessing1.data_mi_selection_export(selected_word_number=i, binary_test_class=0, scale_length=10000,
                                                extract_to_file=True, subfolder='acq'+str(i))

        # earn
        logger.info('preprocessing1 scale_length=10000, binary_test_class=17, selected_word_number=' + str(i))
        preprocessing1.data_mi_selection_export(selected_word_number=i, binary_test_class=17, scale_length=10000,
                                                extract_to_file=True, subfolder='earn'+str(i))

    logger.info('Done !')
if __name__ == "__main__":
    main()