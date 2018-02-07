# MLE for Gaussian Mixture model (GMM)
# Summarize all partial evaluations
#
# @nghia n h | Yamada-lab
# Hard code

"""
# Cause of difference path name rules and simplifying process,
# this script should be only run on unix system and use alphabet folder/file name.

---------------------------------------------------
[sample supervised result file]
---------------------------------------------------
             precision    recall  f1-score   support

      5       0.48      0.37      0.42        35
      6       0.35      0.12      0.17        78
      7       0.27      0.52      0.35       117
      8       0.25      0.41      0.31       170
      9       0.25      0.18      0.21       207
     10       0.18      0.17      0.17       190
     11       0.19      0.22      0.20       146
     12       0.04      0.03      0.03        80
     13       0.18      0.05      0.08        61
     14       0.12      0.05      0.07        38
     15       0.00      0.00      0.00        31

avg / total   0.22      0.23      0.21      1153

#labeled: 404
Acc: 0.2263660017346054

---------------------------------------------------
[sample semi-supervised result file]
---------------------------------------------------
             precision    recall  f1-score   support

          5       0.25      0.80      0.39        35
          6       0.22      0.10      0.14        78
          7       0.25      0.39      0.30       117
          8       0.21      0.21      0.21       170
          9       0.28      0.38      0.32       207
         10       0.11      0.05      0.07       190
         11       0.22      0.23      0.23       146
         12       0.11      0.09      0.10        80
         13       0.33      0.11      0.17        61
         14       0.07      0.03      0.04        38
         15       0.00      0.00      0.00        31

avg / total       0.20      0.22      0.20      1153

#labeled: 672
#unlabeled: 673
Acc: 0.2220294882914137

---------------------------------------------------
The process takes average score on all same files in all folder and return the result in summary folder

"""

import os
import numpy as np

class DataFile(object):
    __doc__ = 'Data file structure'

    def __init__(self, dir, class_number):
        self.dir = dir
        self.file_name = os.path.split(self.dir)[1]
        self.class_number = class_number

        self.data = np.array(['', 'precision', 'recall', ',f1-score', 'support'])
        self.data_info = {'#labeled:':'0', '#unlabeled:':'0', 'Acc:':'0'}
        self.readFile()

    def readFile(self):
        with open(self.dir) as f:
            lines = f.read().splitlines()
            for line in lines[2:2 + self.class_number ]:
                self.data = np.vstack((self.data, np.array(line.split())))
            # todo: avg / total: need to re-caculate
            self.data = np.vstack((self.data, np.array(lines[2 + self.class_number +1].split()[2:])))
            # self.data = np.vstack((self.data, np.zeros(5)))
            self.data[-1,0] = '0.0'

            # #labeled
            for line in lines[16:]:
                splited = line.split()
                self.data_info[splited[0]] = splited[1]


class EvaluationSummary(object):
    __doc__ = 'Summarize all evaluations.' \
              'The partial tests are separated in difference folders and having the same structure' \
              '(number of files, files\'s name)' \


    def __init__(self, map_file, folder_list, report_list):
        self.report_list = report_list
        self.folder_list = folder_list
        self.folder_number = len(folder_list)
        self.report_number = len(report_list)

        with open(map_file) as f:
            self.class_number = len(f.readline().split(','))

    def summarize(self):
        for file in self.report_list:
            sum_data = None
            for folder in self.folder_list:
                data_from_file = DataFile(folder + '/' + file, self.class_number)
                if sum_data is None:
                    sum_data = data_from_file
                else:
                    sum_data.data[1:,1:] = np.sum([sum_data.data[1:,1:].astype(np.float),
                                              data_from_file.data[1:,1:].astype(np.float)], axis=0)
                    sum_data.data_info['Acc:'] = float(sum_data.data_info['Acc:']) + float(data_from_file.data_info['Acc:'])

            # todo: you need to caculate avg / total row
            sum_data.data[1:,1:] = np.divide(sum_data.data[1:,1:].astype(np.float), float(self.folder_number))
            sum_data.data_info['Acc:'] /= float(self.folder_number)

            # write summarized data
            np.savetxt('summary-'+file, sum_data.data[1:,1:].astype(np.float), fmt=['%6.2f']*3 + ['%6.0f'])
            #
            with open('summary-'+file, 'a') as f:
                for key, value in sum_data.data_info.items():
                    f.write(key + str(value) + '\n')


def main():
    # abalone data
    folder_list = ['0','1','2']
    report_list = ['0.3-report', '0.5-report']
    e = EvaluationSummary('data/abalone.map.csv', folder_list, report_list)
    e.summarize()


if __name__ == '__main__':
    main()
