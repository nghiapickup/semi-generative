# semi-supervised learning with Multinomial Naive Bayes model


## How code woks

### [process 1] Reading from formated files
This process requires the pre-processing data first. 
Each data file treats input data as vector of features, each vector per line.

Where train and test files are included label as the last feature of each vector.

The label must be converted into form of continuous non negative number [0, 1, ...] (quite noob here, will fix)

Map file only includes list of origin name for each class 
(just in case the data does not cover all classes)

    Data files(map, train, test)
        --> call Dataset
        - Extract data file. It does not slpit data there.
            --> call SslDataset
            - Split data from Dataset.
            - All splitting schedule for evaluation should be done here.
                --> call training model
                - The interface for each model includes train and test method in basic.

### [process 2] Reading directly from raw data
The only difference with the last process is it does not call Dataset.
Despite that, the process should be inherited data_preprocessing script in Data project.

The only reason is that it saves more time when extract the raw (compressed) data file 
than reading large features vector file.

## Class desciption

### < Dataset >
Generate basic data input format from list file

- __init(*args)__ Init base class or from orther dataset read from args
- __load_from_csv(file_name)__ load data from csv files specified in file_name (train, test, map)

    
    Example file_name
    data-test/map.csv data-test/train.csv data-test/test.csv
    
