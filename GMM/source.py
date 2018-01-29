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
        self.train_unlabeled_file = ''
        self.test_file = ''
        self.problem_type = ''

        # dataset features
        self.train_xl = np.empty((0))
        self.train_yl = np.empty((0))
        self.train_xu = np.empty((0))
        self.test_x = np.empty((0))
        self.test_y = np.empty((0))
        self.class_name = []

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
            train_unlabel_load = np.genfromtxt(self.train_unlabeled_file, delimiter=',')
            self.train_xu = np.mat(train_unlabel_load)

        train_label_load = np.genfromtxt(self.train_label_file, delimiter=',')
        self.train_xl = np.mat(train_label_load.T[:-1].T)
        self.train_yl = np.mat(train_label_load.T[-1])

        test_load = np.genfromtxt(self.test_file, delimiter=',')
        self.test_x = np.mat(test_load.T[:-1].T)
        self.test_y = np.mat(test_load.T[-1])

        map_load = np.genfromtxt(self.map_file, dtype='str', delimiter=',')
        self.class_number = len(map_load)
        self.class_name = map_load

        self.instance_label_number, self.feature_number = np.shape(self.train_xl)
        self.instance_unlabel_number = np.shape(self.train_xu)[0]
        self.instance_test_number = np.shape(self.test_x)[0]


class Evaluation(object):
    __doc__ = 'Result evaluation'

    def __init__(self, label, target, prediction):
        # calculate evaluation index
        self.label = label
        self.accuracy = metrics.accuracy_score(target, prediction)
        self.report = metrics.classification_report(target, prediction, target_names=label)

    def export_report(self, fname):
        with open(fname, 'w') as f:
            f.write(self.report + '\n')
            f.writelines('Acc: ' + str(self.accuracy))


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
        self.predicted_label = np.mat(np.zeros((1,self.data.instance_test_number)))

    def MultivariateGaussian(self, x, mu, sigma):
        p = len(x[0])
        fraction = np.power(2*np.pi,p/2) * np.power(np.linalg.det(sigma), 0.5)
        # notice: put correct parenthesis here
        e_power = float(-0.5*(x-mu) * (np.linalg.pinv(sigma)) * ((x-mu).T))
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
            instance_sum = np.mat(np.zeros((1, self.data.feature_number)))
            for k in range(l):
                if self.data.train_yl[0, k] == i:
                    instance_sum += self.data.train_xl[k]
            self.mu.append(instance_sum/l_j[0,i])

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


class GmmSemisupervised(object):
    __doc__ = 'Deploy GMM model for labeled and unlabeled data D=(Dl,Du) with c classes'


    def __init__(self, dataset):
        self.data = dataset
        # model parameter
        self.pi = []
        self.mu = []
        self.cov = []

        self.loopcount = 0

        # evaluate
        # note: use matrix here, not list for same data type with data.test_y
        self.predicted_label = np.mat(np.zeros((1,self.data.instance_test_number)))

    def MultivariateGaussian(self, x, mu, sigma):
        p = len(x[0])
        fraction = np.power(2*np.pi,p/2) * np.power(np.linalg.det(sigma), 0.5)
        # notice: put correct parenthesis here
        e_power = float(-0.5*(x-mu) * (np.linalg.pinv(sigma)) * ((x-mu).T))
        return np.power(np.e, e_power)/fraction

    def train(self):
        # init parameter
        # using parameter estimated from Gmm supervised
        gmm_all_label = GmmSupervised(self.data)
        gmm_all_label.train()
        self.pi = gmm_all_label.pi
        self.mu = gmm_all_label.mu
        self.cov = gmm_all_label.cov

        epsilon = 1e-3
        diff = 1
        l = self.data.instance_label_number
        u = self.data.instance_unlabel_number

        # EM algorithm
        # dara D = (xl, yl) union (xu)
        while diff > epsilon:
            self.loopcount+=1

            # E step
            pi_old = self.pi
            mu_old = self.mu
            cov_old = self.cov
            self.pi = []
            self.mu = []
            self.cov = []

            # gamma estimate
            gamma = np.mat(np.zeros((l+u,self.data.class_number)))
            # labeled data
            for i in range(self.data.instance_label_number):
                for j in range(self.data.class_number):
                    if self.data.train_yl[0,i] == j:
                        gamma[i,j] = 1
            # unlabeled data
            for i in range(u):
                denominator = 0.
                for j in range(self.data.class_number):
                    gamma[i+l,j] = pi_old[j]*self.MultivariateGaussian(self.data.train_xu[i],mu_old[j],cov_old[j])
                    denominator += gamma[i+l,j] # px = sum over C of P(x|y_c)
                for j in range(self.data.class_number):
                    gamma[i+l,j] /= denominator

            # M step
            # calcute Li
            l_j = np.sum(gamma, axis=0)
            for i in range(self.data.class_number):
                # class proportion
                self.pi.append(l_j[0,i]/(l+u))

                # mean
                instance_sum = np.mat(np.zeros((1, self.data.feature_number)))
                for k in range(l):
                    instance_sum += gamma[k, i]*self.data.train_xl[k]
                for k in range(u):
                    instance_sum += gamma[l+k, i]*self.data.train_xu[k]
                self.mu.append(instance_sum/l_j[0,i])

                # covariance
                cov_sum = np.mat(np.zeros((self.data.feature_number,self.data.feature_number)))
                for k in range(l):
                    cov_sum += gamma[k, i]*((self.data.train_xl[k]-self.mu[i]).T)*(self.data.train_xl[k]-self.mu[i])
                for k in range(u):
                    cov_sum += gamma[l+k, i]*((self.data.train_xu[k]-self.mu[i]).T)*(self.data.train_xu[k]-self.mu[i])
                self.cov.append(cov_sum / l_j[0,i])

            # check convegence of mu
            diff = 0
            for i in range(self.data.class_number):
                diff += ((self.mu[i] - mu_old[i])*(self.mu[i] - mu_old[i]).T)[0]

        print(self.loopcount)

    def test(self):
        # estimated value of x for each class
        estimate_value = np.mat(np.zeros((self.data.instance_test_number, self.data.class_number)))
        for i in range(self.data.instance_test_number):
            for j in range(self.data.class_number):
                estimate_value[i,j] = self.pi[j] * \
                                      self.MultivariateGaussian(self.data.test_x[i],self.mu[j],self.cov[j])
            self.predicted_label[0,i] = np.argmax(estimate_value[i])

        # # calculate evaluation index
        # self.accuracy = metrics.accuracy_score(np.squeeze(np.asarray(self.data.test_y)),
        #                                                   np.squeeze(np.asarray(self.predicted_label)))
        #                 # :( too long to convert back from single matrix to array

# main
def main():
    # Input format
    # <problem type> <map file> <train data, labeled> <train data, unlabeled> <test data>
    #
    #   <problem type> 1: supervised, 2: semi-supervised

    # try:
    # default

    data_file_name = []
    if (len(sys.argv) > 1):
        data_file_name = sys.argv[1:]
    else:
        data_file_name = input("command: ").split()

    # Extract data
    dataset = Dataset()
    dataset.loadCSV(data_file_name)

    # deploy model
    if dataset.problem_type == '1':
        gmm_model = GmmSupervised(dataset)
    elif dataset.problem_type == '2':
        gmm_model = GmmSemisupervised(dataset)

    # learning
    gmm_model.train()
    gmm_model.test()

    e = Evaluation(gmm_model.data.class_name,
                   np.squeeze(np.asarray(gmm_model.data.test_y)),
                   np.squeeze(np.asarray(gmm_model.predicted_label)))
    e.export_report('report')

    # except:
    #     e = sys.exc_info()
    #     print(e[0], e[1])


if __name__ == '__main__':
    main()
    # supervised

    # 1 data/3-5i/iris.map.csv data/3-5i/iris.train.label.csv data/3-5i/iris.test.csv

    # 1 data/5/news.map.csv data/5/news.train.label.csv data/5/news.test.csv

    # 1 data/3-5a/abalone.map.csv data/3-5a/abalone.train.label.csv data/3-5a/abalone.test.csv

    # semi-supervised

    # 2 data/3-5i/iris.map.csv data/3-5i/iris.train.label.csv data/3-5i/iris.train.unlabel.csv data/3-5i/iris.test.csv

    # 2 data/5/news.map.csv data/5/news.train.label.csv data/5/news.train.unlabel.csv data/5/news.test.csv

    # 2 data/3-5a/abalone.map.csv data/3-5a/abalone.train.label.csv data/3-5a/abalone.train.unlabel.csv data/3-5a/abalone.test.csv
