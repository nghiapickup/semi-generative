# semi-supervised learning with GMM model

I implemented GMM model for semi-supervised using Python and numpy.
There are 2 algorithms there: 

- GMM all labeled using derivative solution
- GMM for labeled and unlabeled data using EM algorithm.

Data format could be found here: [Data preprocessing](https://github.com/nghiapickup/Data_for_Semisupervised.git)


## Class desciption

### < Dataset >
Data input format
    <problem type> <map file> <train data, labeled> <train data, unlabeled> <test data>
    
< problem type > 
- 1: supervised data (without unlabeled data)
- 2: semi-supervised data (with unlabeled data)

Example command
    
    1 data/iris.map.csv data/iris.train.label.csv data/iris.test.csv
    2 data/iris.map.csv data/iris.train.label.csv data/iris.train.unlabel.csv data/iris.test.csv

### < GmmSupervised >
GMM model using partial derivatives solution with all labeled data.

The modele is initialized with a < Dataset > instance

### < GmmSemisupervised >
GMM model using EM algorithm with labeled and unlabeled data

The modele is initialized with a < Dataset > instance

### < Evaluation >
Evaluation models
- with cross validation
- export report

### <  >
