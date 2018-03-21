# Data description

## How to run
### < data name > < test proportion > < unlabeled proportion >
Example

    iris 0.3 0.7
    20news 0.4 0.7
    abadone 0.3 0.8

## Input requirement
- Data file
- Lablel file (for checking and re in-dex label in case of train or test data does not cover all classes)

## Output description
#### .train.label.CSV, .train.unlabel.CSV, .test.CSV

Data is splitted into 2 parts train & test (if it doesn't), the proportion should be (nearly) same.
Each part (train or test) has the same CSV data form: vector of features and with corresponding class (except unlabeled data file)
- X and Y are both numeric form.
- Class Y=[0,2,3,...C-1]
    
    
    x_11,x_12,x_13,...,x_1n,y_1
	
	x_21,x_22,x_23,...,x_2n,y_2
	
	...
	
	x_m1,x_m2,x_m3,...,x_mn,y_m

#### .map.CSV>
Map file with all class label meaning.
This also be the based for counting numnber of classes.

    < class name 1 >,< class name 2 >,...,< class name c >
    
## It is worth noting here

- There must be NO space between file name. This help to read input exactly.

- The normal data file is not scale, using scale function to create new scaling data.
It is better to test both scale and not scale in general cases.

- labeled and unlabeled data shoule be in difference files. It is easier to test supervised also.

- Map file for features name is addition option.