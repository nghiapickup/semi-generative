# MLE for Gaussian Mixture model (GMM)
# supervised and semi-supervised
#
# @author: nghianh | Yamada-lab

import sys
import numpy as np
from scipy import stats
from sklearn import model_selection
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
        self.train_yu = np.empty((0)) # so ridiculous :P
        self.test_x = np.empty((0))
        self.test_y = np.empty((0))
        self.class_name = []

        self.class_number = 0
        self.feature_number = 0
        self.instance_label_number = 0
        self.instance_unlabel_number = 0
        self.instance_test_number = 0

    def load_from_CSV(self, file_name):
        self.problem_type = file_name[0]

        if self.problem_type == '1':
            self.map_file, self.train_label_file, self.test_file = file_name[1:]
        else:
            self.map_file, self.train_label_file, self.train_unlabeled_file, self.test_file = file_name[1:]
            # Load unlabeled data
            train_unlabel_load = np.genfromtxt(self.train_unlabeled_file, delimiter=',')
            self.train_xu = np.mat(train_unlabel_load.T[:-1].T)
            self.train_yu = np.mat(train_unlabel_load.T[-1])

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

    def data_from_scaling(self, size):
        # Split labeled and unlabeled data by scaling size[labeled, unlabeled]

        splited_data = Dataset()

        # splitting data, guarantee that number of samples per class are nearly equal

        # labeled data slpiting
        if size[0] == 1:
            splited_data.train_xl = self.train_xl[:]
            splited_data.train_yl = self.train_yl[:]
        else:
            sss1 = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=size[0], random_state=0)
            # notice: set random_state is a constant to make sure that the next scaling is the expand of last data set
            for data_indecices, labeled_indices in sss1.split(self.train_xl, self.train_yl.T):
                splited_data.train_xl = self.train_xl[labeled_indices]
                splited_data.train_yl = self.train_yl[0,labeled_indices]

        # unlabeled data splitting
        if size[1] == 1:
            splited_data.train_xu = self.train_xu
        else:
            sss2 = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=size[1], random_state=0)
            for data_indecices, unlabeled_indices in sss2.split(self.train_xu, self.train_yu.T):
                splited_data.train_xu = self.train_xu[unlabeled_indices]

        # update parameters
        splited_data.problem_type = self.problem_type
        splited_data.test_x = self.test_x
        splited_data.test_y = self.test_y
        splited_data.class_name = self.class_name
        splited_data.class_number = self.class_number
        splited_data.feature_number = self.feature_number
        splited_data.instance_label_number = len(splited_data.train_xl)
        splited_data.instance_unlabel_number = len(splited_data.train_xu)
        splited_data.instance_test_number = self.instance_test_number

        return splited_data

    def data_from_indices_cv(self, train, test):
        # extract data from indices list for cross validation
        # notice here: The test is extracted from train data, not test data of dataset
        splited_data = Dataset()
        splited_data.train_xl = self.train_xl[train]
        splited_data.train_yl = self.train_yl[0,train]
        splited_data.train_xu = self.train_xu[:]
        splited_data.train_yu = self.train_yu[:]
        # notice
        splited_data.test_x = self.train_xl[test]
        splited_data.test_y = self.train_yl[0,test]

        splited_data.problem_type = self.problem_type
        splited_data.class_name = self.class_name
        splited_data.class_number = self.class_number
        splited_data.feature_number = self.feature_number
        splited_data.instance_label_number = len(train)
        splited_data.instance_unlabel_number = self.instance_unlabel_number
        splited_data.instance_test_number = len(test)

        return splited_data

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
        fraction = np.power(2*np.pi,p/2.) * np.power(np.linalg.det(sigma), 0.5)
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
                                      stats.multivariate_normal.pdf(self.data.test_x[i],
                                                                    mean=self.mu[j][0],
                                                                    cov=self.cov[j], allow_singular=True)
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
        fraction = np.power(2*np.pi,p/2.) * np.power(np.linalg.det(sigma), 0.5)
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
                    gamma[i+l,j] = pi_old[j] * stats.multivariate_normal.pdf(self.data.train_xu[i],
                                                                             mean=np.squeeze(np.asarray(mu_old[j])),
                                                                             cov=cov_old[j], allow_singular=True)
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
                ###
                print('MU  ', self.mu[j])
                print('COV  ', self.cov[j])
                ###
                ###
                estimate_value[i,j] = self.pi[j] * \
                                      stats.multivariate_normal.pdf(self.data.test_x[i],
                                                                    mean=self.mu[j][0],
                                                                    cov=self.cov[j],
                                                                    allow_singular=True)
            self.predicted_label[0,i] = np.argmax(estimate_value[i])

class Evaluation(object):
    __doc__ = 'Result evaluation'

    def __init__(self, dataset):
        self.dataset = dataset

    def leave_one_out_cv(self, data, data_model):
        # leave one out cross validation
        # this only

        kf = model_selection.KFold(n_splits=data.instance_label_number)
        pi_cv = None
        mu_cv = None
        cov_cv = None
        for train, test in kf.split(data.train_xl):
            # estimate parameters
            model = data_model(data.data_from_indices_cv(train, test))
            model.train()
            # pi
            if pi_cv is None:
                pi_cv = model.pi
            else:
                pi_cv = np.sum([pi_cv, model.pi], axis=0)
            # mu:
            if mu_cv is None:
                mu_cv = model.mu
            else:
                mu_cv = np.sum([mu_cv, model.mu], axis=0)
            # cov
            if cov_cv is None:
                cov_cv = model.cov
            else:
                cov_cv = np.sum([cov_cv, model.cov], axis=0)

        pi_cv = np.divide(pi_cv, float(data.instance_label_number))
        mu_cv = np.divide(mu_cv, float(data.instance_label_number))
        cov_cv = np.divide(cov_cv, float(data.instance_label_number))

        # test with only labeled data
        model = data_model(data)
        model.pi = pi_cv
        model.mu = mu_cv
        model.cov = cov_cv

        model.test()
        return model

    def report_export(self, model, fname, mode=1):
        label = model.data.class_name
        target = np.squeeze(np.asarray(model.data.test_y))
        prediction = np.squeeze(np.asarray(model.predicted_label))

        # calculate evaluation index
        accuracy = metrics.accuracy_score(target, prediction)
        report = metrics.classification_report(target, prediction, target_names=label)

        # export report
        with open(fname, 'w') as f:
            f.write(report)
            f.writelines('\n#labeled: ' + str(model.data.instance_label_number))
            if(mode == 2):
                f.writelines('\n#unlabeled: ' + str(model.data.instance_unlabel_number))
            f.writelines('\nAcc: ' + str(accuracy))

    def abalone_test(self):
        label_scaling = (0.3, 0.5, 0.7, 1.0)
        unlabel_scaling = (0.3, 0.5, 0.7, 1.0)
        for i in label_scaling:
            scaled_data = Dataset()
            for j in unlabel_scaling:
                # scale data first
                scaled_data = self.dataset.data_from_scaling([i, j])
                # semi-supervised
                gmm_model = self.leave_one_out_cv(scaled_data, GmmSemisupervised)
                report_file_name = str(i) + '_' + str(j) + '-report'
                self.report_export(gmm_model, report_file_name, 2)

            # supervised
            gmm_model = self.leave_one_out_cv(scaled_data, GmmSupervised)
            report_file_name = str(i) + '-report'
            self.report_export(gmm_model, report_file_name)

# main
def main():
    # Input format
    # <problem type> <map file> <train data, labeled> <train data, unlabeled> <test data>
    #
    #   <problem type> 1: supervised, 2: semi-supervised

    # try:
    # default

    if (len(sys.argv) > 1):
        data_file_name = sys.argv[1:]
    else:
        data_file_name = input("command: ").split()

    # Extract data
    dataset = Dataset()
    dataset.load_from_CSV(data_file_name)

    e = Evaluation(dataset)
    e.abalone_test()

    print('Done')

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
