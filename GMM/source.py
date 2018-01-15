# MLE for Gaussian Mixture model (GMM)
# supervised and semi-supervised
#
# @author: nghianh | Yamada-lab

import sys
import numpy as np
from sklearn import metrics


class Dataset(object):
    __doc__ = 'Handle data frame'

    def __init__(self):
        self.map_file = ''
        self.train_labeled_file = ''
        self.train_unlabel_file = ''
        self.test_file = ''
        self.problem_type = ''

        # dataset features
        self.train_xl = np.empty((0))
        self.train_yl = np.empty((0))
        self.train_xu = np.empty((0))
        self.test_x = np.empty((0))
        self.test_y = np.empty((0))

        self.class_number = 0
        self.feature_number = 0
        self.instance_label_number = 0
        self.instance_unlabel_number = 0
        self.instance_test_number = 0

    def loadCSV(self, file_name):
        self.problem_type = file_name[0]

        if self.problem_type == '1':
            self.map_file, self.train_label_file, self.test_file = file_name[1:]
        else:
            self.map_file, self.train_label_file, self.train_unlabeled_file, self.test_file = file_name[1:]
            # Load unlabeled data
            train_unlabel_load = np.genfromtxt(self.train_unlabel_file, delimiter=',')
            self.train_xu = np.mat(train_unlabel_load.T[:-1].T)

        train_label_load = np.genfromtxt(self.train_label_file, delimiter=',')
        self.train_xl = np.mat(train_label_load.T[:-1].T)
        self.train_yl = np.mat(train_label_load.T[-1])

        test_load = np.genfromtxt(self.test_file, delimiter=',')
        self.test_x = np.mat(test_load.T[:-1].T)
        self.test_y = np.mat(test_load.T[-1])

        map_load = np.genfromtxt(self.map_file, delimiter=',')
        self.class_number = int(map_load[-1])

        self.instance_label_number, self.feature_number = np.shape(self.train_xl)
        self.instance_unlabel_number = np.shape(self.train_xu)[0]
        self.instance_test_number = np.shape(self.test_x)[0]

class GmmSupervised(object):
    __doc__ = 'Deploy GMM model for fully labeled data D=(X,Y) with c classes'

    def __init__(self, dataset):
        self.data = dataset
        # model parameter
        self.pi = []
        self.mu = []
        self.cov = []

        # evaluate
        # note: use matrix here, not list for same data type with data.test_y
        self.predicted_label = np.mat(np.zeros((1,se lf.data.instance_test_number)))
        self.accuracy = .0

    def MultivariateGaussian(self, x, mu, sigma):
        p = len(x[0])
        fraction = np.power(2*np.pi,p/2) * np.power(np.linalg.det(sigma), 0.5)
        # notice: put correct parenthesis here
        e_power = float(-0.5*(x-mu) * (np.linalg.inv(sigma)) * ((x-mu).T))
        return np.power(np.e, e_power)/fraction

    def train(self):
        # calcute Li
        l = self.data.instance_label_number
        l_j = np.mat(np.zeros((1,self.data.class_number)))
        for i in range(l):
            l_j[0,int(self.data.train_yl[0,i])] += 1

        # estimate parameters
        for i in range(self.data.class_number):
            # class proportion
            self.pi.append(l_j[0,i]/l)

            # mean
            class_sum = np.mat(np.zeros((1, self.data.feature_number)))
            for k in range(l):
                if self.data.train_yl[0, k] == i:
                    class_sum += self.data.train_xl[k]
            self.mu.append(class_sum/l_j[0,i])

            # covariance
            cov_sum = np.mat(np.zeros((self.data.feature_number,self.data.feature_number)))
            for k in range(l):
                if self.data.train_yl[0,k] == i:
                    cov_sum += ((self.data.train_xl[k]-self.mu[i]).T)*(self.data.train_xl[k]-self.mu[i])
            self.cov.append(cov_sum / l_j[0,i])

    def test(self):
        # estimated value of x for each class
        estimate_value = np.mat(np.zeros((self.data.instance_test_number, self.data.class_number)))
        for i in range(self.data.instance_test_number):
            for j in range(self.data.class_number):
                estimate_value[i,j] = self.pi[j] * \
                                      self.MultivariateGaussian(self.data.test_x[i],self.mu[j],self.cov[j])
            self.predicted_label[0,i] = np.argmax(estimate_value[i])

        # calculate evaluation index
        self.accuracy = metrics.accuracy_score(np.squeeze(np.asarray(self.data.test_y)),
                                                          np.squeeze(np.asarray(self.predicted_label)))
                        # :( too long to convert back from single matrix to array
        print(self.accuracy)


class GmmSemisupervised(object):
    __doc__ = 'Deploy GMM model for labeled and unlabeled data D=(Dl,Du) with c classes'


    def __init__(self, dataset):
        self.data = dataset
        # model parameter
        self.pi = []
        self.mu = []
        self.cov = []

        # evaluate
        # note: use matrix here, not list for same data type with data.test_y
        self.predicted_label = np.mat(np.zeros((1,self.data.instance_test_number)))
        self.accuracy = .0

    def MultivariateGaussian(self, x, mu, sigma):
        p = len(x[0])
        fraction = np.power(2*np.pi,p/2) * np.power(np.linalg.det(sigma), 0.5)
        # notice: put correct parenthesis here
        e_power = float(-0.5*(x-mu) * (np.linalg.inv(sigma)) * ((x-mu).T))
        return np.power(np.e, e_power)/fraction

    def train(self):
        # calcute Li
        l = self.data.instance_label_number
        l_j = np.mat(np.zeros((1,self.data.class_number)))
        for i in range(l):
            l_j[0,int(self.data.train_yl[0,i])] += 1

        # estimate parameters
        for i in range(self.data.class_number):
            # class proportion
            self.pi.append(l_j[0,i]/l)

            # mean
            class_sum = np.mat(np.zeros((1, self.data.feature_number)))
            for k in range(l):
                if self.data.train_yl[0, k] == i:
                    class_sum += self.data.train_xl[k]
            self.mu.append(class_sum/l_j[0,i])

            # covariance
            cov_sum = np.mat(np.zeros((self.data.feature_number,self.data.feature_number)))
            for k in range(l):
                if self.data.train_yl[0,k] == i:
                    cov_sum += ((self.data.train_xl[k]-self.mu[i]).T)*(self.data.train_xl[k]-self.mu[i])
            self.cov.append(cov_sum / l_j[0,i])

    def test(self):
        # estimated value of x for each class
        estimate_value = np.mat(np.zeros((self.data.instance_test_number, self.data.class_number)))
        for i in range(self.data.instance_test_number):
            for j in range(self.data.class_number):
                estimate_value[i,j] = self.pi[j] * \
                                      self.MultivariateGaussian(self.data.test_x[i],self.mu[j],self.cov[j])
            self.predicted_label[0,i] = np.argmax(estimate_value[i])

        # calculate evaluation index
        self.accuracy = metrics.accuracy_score(np.squeeze(np.asarray(self.data.test_y)),
                                                          np.squeeze(np.asarray(self.predicted_label)))
                        # :( too long to convert back from single matrix to array
        print(self.accuracy)


# main
def main():
    # Input format
    # <problem type> <map file> <train data, labeled> <train data, unlabeled> <test data>
    #
    ## <problem type>
    ## 1: supervised, 2: semi-supervised
    # try:
    # default
    data_file_name = []

    if (len(sys.argv) > 1):
        # todo utilize terminal/cmd
        print('')
    else:
        data_file_name = input("command: ").split()

    # Extract data
    dataset = Dataset()
    dataset.loadCSV(data_file_name)

    # deploy model
    if dataset.problem_type == '1':
        gmm_model = GmmSupervised(dataset)
    else:
        gmm_model = GmmSemisupervised(dataset)

    # learning
    gmm_model.train()
    gmm_model.test()
    # except:
    #     e = sys.exc_info()[0]
    #     print(e)


if __name__ == '__main__':
    main()
    # iris data
    # 1 data/iris.map.csv data/iris.train.csv data/iris.test.csv
