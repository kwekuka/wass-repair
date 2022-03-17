# wass-repair


The main goal of this code is to show how to transform distributions in the Wasserstein space. The method which achieves this goal is called `bcmap.geometric_adjustment`. 

Some important notes before using the code
* the methods assume your data (training/testing) is input as a `pandas.dataframe`
* CDFs are a common object in the code, and are represented as discrete empirical CDFs via `np.ndarry`.  This means that given continuous scores between 0 and 1 (eg. 0.07, .32, .9, etc) that we must also provide the bins which will discretize the scores. So for example, if we provide bins `[.33, .66, 1]`, then the vector representation of the scores given previously would be `[2, 0, 1]` with a CDF representation `[2/3,2/3,1]`.  For binary classification and regression, having equal width bins between the min score and the max score (usually 100 bins between 0 and 1) is sufficient. 


## Preparing Your Data 
In order to pass a dataframe to the geometric repair method, it must contain three columns.  Each individual will correspond to a row.  In each row, the three columns are :
* **score** - this contains the score denoting the likelihood of the positive classifcation instance. will be task specific but its likely a number between 0 << 1 
* **label** a 0/1 flag denoting if this item belongs to the positive class in the dataset 
* **group** a 0/1 flag denoting the protected group membership 

### A note on Train/Test Split 
Here, the naming convention of training data vs. testing data is simply used to describe the data that are used to determine/compute the reparied scores for propostprocessing (training data) and the dataframe that the repair will be applied onto (testing data).  Unlike other machine learning applications, these data may be the same, or may be different.  The dataframe that is returned is the testing data with the postprocessing applied to it. 
