# Population-Segmentation
Population_Segmentation with Sagemaker



Population Segmentation with SageMaker
 Employing two unsupervised learning algorithms to do population segmentation. Population segmentation aims to find natural groupings in population data that reveal some feature-level similarities between different regions in the US.

Using principal component analysis (PCA) we reduce the dimensionality of the original census data. Then, we use k-means clustering to assign each US county to a particular cluster based on where a county lies in component space. How each cluster is arranged in component space can tell you which US counties are most similar and what demographic traits define that similarity; this information is most often used to inform targeted, marketing campaigns that want to appeal to a specific group of people. This cluster information is also useful for learning more about a population by revealing patterns between regions that you otherwise may not have noticed.

US Census Data

Machine Learning Workflow
To implement population segmentation, you'll go through a number of steps:

Data loading and exploration
Data cleaning and pre-processing
Dimensionality reduction with PCA
Feature engineering and data transformation
Clustering transformed data with k-means
Extracting trained model attributes and visualizing k clusters
These tasks make up a complete, machine learning workflow from data loading and cleaning to model deployment. Each exercise is designed to give you practice with part of the machine learning workflow, and to demonstrate how to use SageMaker tools, such as built-in data management with S3 and built-in algorithms.

First, import the relevant libraries into this SageMaker notebook.

In [2]:
# data managing and display libs
import pandas as pd
import numpy as np
import os
import io

import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline
In [3]:
# sagemaker libraries
import boto3
import sagemaker
Loading the Data from Amazon S3
This particular dataset is already in an Amazon S3 bucket; you can load the data by pointing to this bucket and getting a data file by name.

You can interact with S3 using a boto3 client.

In [4]:
# boto3 client to get S3 data
s3_client = boto3.client('s3')
# S3 bucket name
bucket_name='aws-ml-blog-sagemaker-census-segmentation'
Take a look at the contents of this bucket; get a list of objects that are contained within the bucket and print out the names of the objects. You should see that there is one file, 'Census_Data_for_SageMaker.csv'.

In [5]:
# get a list of objects in the bucket
obj_list=s3_client.list_objects(Bucket=bucket_name)

# print object(s)in S3 bucket
files=[]
for contents in obj_list['Contents']:
    files.append(contents['Key'])
    
print(files)
['Census_Data_for_SageMaker.csv']
In [6]:
# there is one file --> one key
file_name=files[0]

print(file_name)
Census_Data_for_SageMaker.csv
Retrieve the data file from the bucket with a call to client.get_object().

In [7]:
# get an S3 object by passing in the bucket and file name
data_object = s3_client.get_object(Bucket=bucket_name, Key=file_name)

# what info does the object contain?
display(data_object)
{'ResponseMetadata': {'RequestId': 'BDF2626E80A9D106',
  'HostId': 'rJGVoQcGEgWO1A0qgYIhadqWBWjAf/9FHxuai6IObBFm++9aC+sbKm/FqeQhJs2X8XRb+f2aYvc=',
  'HTTPStatusCode': 200,
  'HTTPHeaders': {'x-amz-id-2': 'rJGVoQcGEgWO1A0qgYIhadqWBWjAf/9FHxuai6IObBFm++9aC+sbKm/FqeQhJs2X8XRb+f2aYvc=',
   'x-amz-request-id': 'BDF2626E80A9D106',
   'date': 'Fri, 08 Mar 2019 00:07:34 GMT',
   'last-modified': 'Wed, 12 Sep 2018 15:13:37 GMT',
   'etag': '"066d37f43f7762f1eb409b1660fe9763"',
   'accept-ranges': 'bytes',
   'content-type': 'text/csv',
   'content-length': '613237',
   'server': 'AmazonS3'},
  'RetryAttempts': 0},
 'AcceptRanges': 'bytes',
 'LastModified': datetime.datetime(2018, 9, 12, 15, 13, 37, tzinfo=tzutc()),
 'ContentLength': 613237,
 'ETag': '"066d37f43f7762f1eb409b1660fe9763"',
 'ContentType': 'text/csv',
 'Metadata': {},
 'Body': <botocore.response.StreamingBody at 0x7fc8c171fc18>}
In [8]:
# information is in the "Body" of the object
data_body = data_object["Body"].read()
print('Data type: ', type(data_body))
Data type:  <class 'bytes'>
This is a bytes datatype, which you can read it in using io.BytesIO(file).

In [9]:
# read in bytes data
data_stream = io.BytesIO(data_body)

# create a dataframe
counties_df = pd.read_csv(data_stream, header=0, delimiter=",") 
counties_df.head()
Out[9]:
CensusId	State	County	TotalPop	Men	Women	Hispanic	White	Black	Native	...	Walk	OtherTransp	WorkAtHome	MeanCommute	Employed	PrivateWork	PublicWork	SelfEmployed	FamilyWork	Unemployment
0	1001	Alabama	Autauga	55221	26745	28476	2.6	75.8	18.5	0.4	...	0.5	1.3	1.8	26.5	23986	73.6	20.9	5.5	0.0	7.6
1	1003	Alabama	Baldwin	195121	95314	99807	4.5	83.1	9.5	0.6	...	1.0	1.4	3.9	26.4	85953	81.5	12.3	5.8	0.4	7.5
2	1005	Alabama	Barbour	26932	14497	12435	4.6	46.2	46.7	0.2	...	1.8	1.5	1.6	24.1	8597	71.8	20.8	7.3	0.1	17.6
3	1007	Alabama	Bibb	22604	12073	10531	2.2	74.5	21.4	0.4	...	0.6	1.5	0.7	28.8	8294	76.8	16.1	6.7	0.4	8.3
4	1009	Alabama	Blount	57710	28512	29198	8.6	87.9	1.5	0.3	...	0.9	0.4	2.3	34.9	22189	82.0	13.5	4.2	0.4	7.7
5 rows × 37 columns

Exploratory Data Analysis (EDA)
Now that you've loaded in the data, it is time to clean it up, explore it, and pre-process it. Data exploration is one of the most important parts of the machine learning workflow because it allows you to notice any initial patterns in data distribution and features that may inform how you proceed with modeling and clustering the data.

EXERCISE: Explore data & drop any incomplete rows of data
When you first explore the data, it is good to know what you are working with. How many data points and features are you starting with, and what kind of information can you get at a first glance? In this notebook, you're required to use complete data points to train a model. So, your first exercise will be to investigate the shape of this data and implement a simple, data cleaning step: dropping any incomplete rows of data.

You should be able to answer the question: How many data points and features are in the original, provided dataset? (And how many points are left after dropping any incomplete rows?)

In [10]:
# print out stats about data
# rows = data, cols = features
print('(orig) rows, cols: ', counties_df.shape)

# drop any incomplete data
clean_counties_df = counties_df.dropna(axis=0)
print('(clean) rows, cols: ', clean_counties_df.shape)
(orig) rows, cols:  (3220, 37)
(clean) rows, cols:  (3218, 37)
EXERCISE: Create a new DataFrame, indexed by 'State-County'
Eventually, you'll want to feed these features into a machine learning model. Machine learning models need numerical data to learn from and not categorical data like strings (State, County). So, you'll reformat this data such that it is indexed by region and you'll also drop any features that are not useful for clustering.

To complete this task, perform the following steps, using your clean DataFrame, generated above:

Combine the descriptive columns, 'State' and 'County', into one, new categorical column, 'State-County'.
Index the data by this unique State-County name.
After doing this, drop the old State and County columns and the CensusId column, which does not give us any meaningful demographic information.
After completing this task, you should have a DataFrame with 'State-County' as the index, and 34 columns of numerical data for each county. You should get a resultant DataFrame that looks like the following (truncated for display purposes):

                TotalPop     Men      Women    Hispanic    ...

Alabama-Autauga    55221     26745    28476    2.6         ...
Alabama-Baldwin    195121    95314    99807    4.5         ...
Alabama-Barbour    26932     14497    12435    4.6         ...
...
In [11]:
# index data by 'State-County'
clean_counties_df.index=clean_counties_df['State'] + "-" + clean_counties_df['County']
clean_counties_df.head()
Out[11]:
CensusId	State	County	TotalPop	Men	Women	Hispanic	White	Black	Native	...	Walk	OtherTransp	WorkAtHome	MeanCommute	Employed	PrivateWork	PublicWork	SelfEmployed	FamilyWork	Unemployment
Alabama-Autauga	1001	Alabama	Autauga	55221	26745	28476	2.6	75.8	18.5	0.4	...	0.5	1.3	1.8	26.5	23986	73.6	20.9	5.5	0.0	7.6
Alabama-Baldwin	1003	Alabama	Baldwin	195121	95314	99807	4.5	83.1	9.5	0.6	...	1.0	1.4	3.9	26.4	85953	81.5	12.3	5.8	0.4	7.5
Alabama-Barbour	1005	Alabama	Barbour	26932	14497	12435	4.6	46.2	46.7	0.2	...	1.8	1.5	1.6	24.1	8597	71.8	20.8	7.3	0.1	17.6
Alabama-Bibb	1007	Alabama	Bibb	22604	12073	10531	2.2	74.5	21.4	0.4	...	0.6	1.5	0.7	28.8	8294	76.8	16.1	6.7	0.4	8.3
Alabama-Blount	1009	Alabama	Blount	57710	28512	29198	8.6	87.9	1.5	0.3	...	0.9	0.4	2.3	34.9	22189	82.0	13.5	4.2	0.4	7.7
5 rows × 37 columns

In [12]:
# drop the old State and County columns, and the CensusId column
# clean df should be modified or created anew
drop=["CensusId" , "State" , "County"]
clean_counties_df = clean_counties_df.drop(columns=drop)
clean_counties_df.head()
Out[12]:
TotalPop	Men	Women	Hispanic	White	Black	Native	Asian	Pacific	Citizen	...	Walk	OtherTransp	WorkAtHome	MeanCommute	Employed	PrivateWork	PublicWork	SelfEmployed	FamilyWork	Unemployment
Alabama-Autauga	55221	26745	28476	2.6	75.8	18.5	0.4	1.0	0.0	40725	...	0.5	1.3	1.8	26.5	23986	73.6	20.9	5.5	0.0	7.6
Alabama-Baldwin	195121	95314	99807	4.5	83.1	9.5	0.6	0.7	0.0	147695	...	1.0	1.4	3.9	26.4	85953	81.5	12.3	5.8	0.4	7.5
Alabama-Barbour	26932	14497	12435	4.6	46.2	46.7	0.2	0.4	0.0	20714	...	1.8	1.5	1.6	24.1	8597	71.8	20.8	7.3	0.1	17.6
Alabama-Bibb	22604	12073	10531	2.2	74.5	21.4	0.4	0.1	0.0	17495	...	0.6	1.5	0.7	28.8	8294	76.8	16.1	6.7	0.4	8.3
Alabama-Blount	57710	28512	29198	8.6	87.9	1.5	0.3	0.1	0.0	42345	...	0.9	0.4	2.3	34.9	22189	82.0	13.5	4.2	0.4	7.7
5 rows × 34 columns

Now, what features do you have to work with?

In [13]:
# features
features_list = clean_counties_df.columns.values
print('Features: \n', features_list)
Features: 
 ['TotalPop' 'Men' 'Women' 'Hispanic' 'White' 'Black' 'Native' 'Asian'
 'Pacific' 'Citizen' 'Income' 'IncomeErr' 'IncomePerCap' 'IncomePerCapErr'
 'Poverty' 'ChildPoverty' 'Professional' 'Service' 'Office' 'Construction'
 'Production' 'Drive' 'Carpool' 'Transit' 'Walk' 'OtherTransp'
 'WorkAtHome' 'MeanCommute' 'Employed' 'PrivateWork' 'PublicWork'
 'SelfEmployed' 'FamilyWork' 'Unemployment']
Visualizing the Data
In general, you can see that features come in a variety of ranges, mostly percentages from 0-100, and counts that are integer values in a large range. Let's visualize the data in some of our feature columns and see what the distribution, over all counties, looks like.

The below cell displays histograms, which show the distribution of data points over discrete feature ranges. The x-axis represents the different bins; each bin is defined by a specific range of values that a feature can take, say between the values 0-5 and 5-10, and so on. The y-axis is the frequency of occurrence or the number of county data points that fall into each bin. I find it helpful to use the y-axis values for relative comparisons between different features.

Below, I'm plotting a histogram comparing methods of commuting to work over all of the counties. I just copied these feature names from the list of column names, printed above. I also know that all of these features are represented as percentages (%) in the original data, so the x-axes of these plots will be comparable.

In [14]:
# transportation (to work)
transport_list = ['Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp']
n_bins = 50 # can decrease to get a wider bin (or vice versa)

for column_name in transport_list:
    ax=plt.subplots(figsize=(6,3))
    # get data by column_name and display a histogram
    ax = plt.hist(clean_counties_df[column_name], bins=n_bins)
    title="Histogram of " + column_name
    plt.title(title, fontsize=12)
    plt.show()





EXERCISE: Create histograms of your own
Commute transportation method is just one category of features. If you take a look at the 34 features, you can see data on profession, race, income, and more. Display a set of histograms that interest you!

In [15]:
# create a list of features that you want to compare or examine
# employment types
my_list = ['PrivateWork', 'PublicWork', 'SelfEmployed', 'FamilyWork', 'Unemployment']
n_bins = 30 # define n_bins

# histogram creation code is similar to above
for column_name in my_list:
    ax=plt.subplots(figsize=(6,3))
    # get data by column_name and display a histogram
    ax = plt.hist(clean_counties_df[column_name], bins=n_bins)
    title="Histogram of " + column_name
    plt.title(title, fontsize=12)
    plt.show()





EXERCISE: Normalize the data
You need to standardize the scale of the numerical columns in order to consistently compare the values of different features. You can use a MinMaxScaler to transform the numerical values so that they all fall between 0 and 1.

In [16]:
# scale numerical features into a normalized range, 0-1

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
# store them in this dataframe
counties_scaled=pd.DataFrame(scaler.fit_transform(clean_counties_df.astype(float)))

# get same features and State-County indices
counties_scaled.columns=clean_counties_df.columns
counties_scaled.index=clean_counties_df.index

counties_scaled.head()
Out[16]:
TotalPop	Men	Women	Hispanic	White	Black	Native	Asian	Pacific	Citizen	...	Walk	OtherTransp	WorkAtHome	MeanCommute	Employed	PrivateWork	PublicWork	SelfEmployed	FamilyWork	Unemployment
Alabama-Autauga	0.005475	0.005381	0.005566	0.026026	0.759519	0.215367	0.004343	0.024038	0.0	0.006702	...	0.007022	0.033248	0.048387	0.552430	0.005139	0.750000	0.250000	0.150273	0.000000	0.208219
Alabama-Baldwin	0.019411	0.019246	0.019572	0.045045	0.832665	0.110594	0.006515	0.016827	0.0	0.024393	...	0.014045	0.035806	0.104839	0.549872	0.018507	0.884354	0.107616	0.158470	0.040816	0.205479
Alabama-Barbour	0.002656	0.002904	0.002416	0.046046	0.462926	0.543655	0.002172	0.009615	0.0	0.003393	...	0.025281	0.038363	0.043011	0.491049	0.001819	0.719388	0.248344	0.199454	0.010204	0.482192
Alabama-Bibb	0.002225	0.002414	0.002042	0.022022	0.746493	0.249127	0.004343	0.002404	0.0	0.002860	...	0.008427	0.038363	0.018817	0.611253	0.001754	0.804422	0.170530	0.183060	0.040816	0.227397
Alabama-Blount	0.005722	0.005738	0.005707	0.086086	0.880762	0.017462	0.003257	0.002404	0.0	0.006970	...	0.012640	0.010230	0.061828	0.767263	0.004751	0.892857	0.127483	0.114754	0.040816	0.210959
5 rows × 34 columns

In [17]:
counties_scaled.describe()
Out[17]:
TotalPop	Men	Women	Hispanic	White	Black	Native	Asian	Pacific	Citizen	...	Walk	OtherTransp	WorkAtHome	MeanCommute	Employed	PrivateWork	PublicWork	SelfEmployed	FamilyWork	Unemployment
count	3218.000000	3218.000000	3218.000000	3218.000000	3218.000000	3218.000000	3218.000000	3218.000000	3218.000000	3218.000000	...	3218.000000	3218.000000	3218.000000	3218.000000	3218.000000	3218.000000	3218.000000	3218.000000	3218.000000	3218.000000
mean	0.009883	0.009866	0.009899	0.110170	0.756024	0.100942	0.018682	0.029405	0.006470	0.011540	...	0.046496	0.041154	0.124428	0.470140	0.009806	0.760810	0.194426	0.216744	0.029417	0.221775
std	0.031818	0.031692	0.031948	0.192617	0.229682	0.166262	0.078748	0.062744	0.035446	0.033933	...	0.051956	0.042321	0.085301	0.143135	0.032305	0.132949	0.106923	0.106947	0.046451	0.112138
min	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	...	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
25%	0.001092	0.001117	0.001069	0.019019	0.642285	0.005821	0.001086	0.004808	0.000000	0.001371	...	0.019663	0.023018	0.072581	0.373402	0.000948	0.697279	0.120861	0.147541	0.010204	0.150685
50%	0.002571	0.002591	0.002539	0.039039	0.842685	0.022119	0.003257	0.012019	0.000000	0.003219	...	0.033708	0.033248	0.104839	0.462916	0.002234	0.785714	0.172185	0.188525	0.020408	0.208219
75%	0.006594	0.006645	0.006556	0.098098	0.933868	0.111758	0.006515	0.028846	0.000000	0.008237	...	0.056180	0.048593	0.150538	0.560102	0.006144	0.853741	0.243377	0.256831	0.030612	0.271233
max	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	...	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000
8 rows × 34 columns

Data Modeling
Now, the data is ready to be fed into a machine learning model!

Each data point has 34 features, which means the data is 34-dimensional. Clustering algorithms rely on finding clusters in n-dimensional feature space. For higher dimensions, an algorithm like k-means has a difficult time figuring out which features are most important, and the result is, often, noisier clusters.

Some dimensions are not as important as others. For example, if every county in our dataset has the same rate of unemployment, then that particular feature doesn’t give us any distinguishing information; it will not help t separate counties into different groups because its value doesn’t vary between counties.

Instead, we really want to find the features that help to separate and group data. We want to find features that cause the most variance in the dataset!

So, before I cluster this data, I’ll want to take a dimensionality reduction step. My aim will be to form a smaller set of features that will better help to separate our data. The technique I’ll use is called PCA or principal component analysis

Dimensionality Reduction
PCA attempts to reduce the number of features within a dataset while retaining the “principal components”, which are defined as weighted, linear combinations of existing features that are designed to be linearly independent and account for the largest possible variability in the data! You can think of this method as taking many features and combining similar or redundant features together to form a new, smaller feature set.

We can reduce dimensionality with the built-in SageMaker model for PCA.

Roles and Buckets
To create a model, you'll first need to specify an IAM role, and to save the model attributes, you'll need to store them in an S3 bucket.

The get_execution_role function retrieves the IAM role you created at the time you created your notebook instance. Roles are essentially used to manage permissions and you can read more about that in this documentation. For now, know that we have a FullAccess notebook, which allowed us to access and download the census data stored in S3.

You must specify a bucket name for an S3 bucket in your account where you want SageMaker model parameters to be stored. Note that the bucket must be in the same region as this notebook. You can get a default S3 bucket, which automatically creates a bucket for you and in your region, by storing the current SageMaker session and calling session.default_bucket().

In [18]:
from sagemaker import get_execution_role

session = sagemaker.Session() # store the current SageMaker session

# get IAM role
role = get_execution_role()
print(role)
arn:aws:iam::467380521728:role/service-role/AmazonSageMaker-ExecutionRole-20190307T133365
In [19]:
# get default bucket
bucket_name = session.default_bucket()
print(bucket_name)
print()
sagemaker-us-west-1-467380521728

Define a PCA Model
To create a PCA model, I'll use the built-in SageMaker resource. A SageMaker estimator requires a number of parameters to be specified; these define the type of training instance to use and the model hyperparameters. A PCA model requires the following constructor arguments:

role: The IAM role, which was specified, above.
train_instance_count: The number of training instances (typically, 1).
train_instance_type: The type of SageMaker instance for training.
num_components: An integer that defines the number of PCA components to produce.
sagemaker_session: The session used to train on SageMaker.
Documentation on the PCA model can be found here.

Below, I first specify where to save the model training data, the output_path.

In [20]:
# define location to store model artifacts
prefix = 'counties'

output_path='s3://{}/{}/'.format(bucket_name, prefix)

print('Training artifacts will be uploaded to: {}'.format(output_path))
Training artifacts will be uploaded to: s3://sagemaker-us-west-1-467380521728/counties/
In [21]:
# define a PCA model
from sagemaker import PCA

# this is current features - 1
# you'll select only a portion of these to use, later
N_COMPONENTS=33

pca_SM = PCA(role=role,
             train_instance_count=1,
             train_instance_type='ml.c4.xlarge',
             output_path=output_path, # specified, above
             num_components=N_COMPONENTS, 
             sagemaker_session=session)
Convert data into a RecordSet format
Next, prepare the data for a built-in model by converting the DataFrame to a numpy array of float values.

The record_set function in the SageMaker PCA model converts a numpy array into a RecordSet format that is the required format for the training input data. This is a requirement for all of SageMaker's built-in models. The use of this data type is one of the reasons that allows training of models within Amazon SageMaker to perform faster, especially for large datasets.

In [22]:
# convert df to np array
train_data_np = counties_scaled.values.astype('float32')

# convert to RecordSet format
formatted_train_data = pca_SM.record_set(train_data_np)
Train the model
Call the fit function on the PCA model, passing in our formatted, training data. This spins up a training instance to perform the training job.

Note that it takes the longest to launch the specified training instance; the fitting itself doesn't take much time.

In [23]:
%%time

# train the PCA mode on the formatted data
pca_SM.fit(formatted_train_data)
INFO:sagemaker:Creating training-job with name: pca-2019-03-08-00-07-38-756
2019-03-08 00:07:38 Starting - Starting the training job...
2019-03-08 00:07:41 Starting - Launching requested ML instances......
2019-03-08 00:08:42 Starting - Preparing the instances for training......
2019-03-08 00:10:05 Downloading - Downloading input data..
Docker entrypoint called with argument(s): train
[03/08/2019 00:10:18 INFO 140212411717440] Reading default configuration from /opt/amazon/lib/python2.7/site-packages/algorithm/resources/default-conf.json: {u'_num_gpus': u'auto', u'_log_level': u'info', u'subtract_mean': u'true', u'force_dense': u'true', u'epochs': 1, u'algorithm_mode': u'regular', u'extra_components': u'-1', u'_kvstore': u'dist_sync', u'_num_kv_servers': u'auto'}
[03/08/2019 00:10:18 INFO 140212411717440] Reading provided configuration from /opt/ml/input/config/hyperparameters.json: {u'feature_dim': u'34', u'mini_batch_size': u'500', u'num_components': u'33'}
[03/08/2019 00:10:18 INFO 140212411717440] Final configuration: {u'num_components': u'33', u'_num_gpus': u'auto', u'_log_level': u'info', u'subtract_mean': u'true', u'force_dense': u'true', u'epochs': 1, u'algorithm_mode': u'regular', u'feature_dim': u'34', u'extra_components': u'-1', u'_kvstore': u'dist_sync', u'_num_kv_servers': u'auto', u'mini_batch_size': u'500'}
[03/08/2019 00:10:18 WARNING 140212411717440] Loggers have already been setup.
[03/08/2019 00:10:18 INFO 140212411717440] Launching parameter server for role scheduler
[03/08/2019 00:10:18 INFO 140212411717440] {'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/2a343c78-c2c7-48e8-bf1d-79ad865e5656', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'SAGEMAKER_HTTP_PORT': '8080', 'HOME': '/root', 'PYTHONUNBUFFERED': 'TRUE', 'CANONICAL_ENVROOT': '/opt/amazon', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python2.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'LANG': 'en_US.utf8', 'DMLC_INTERFACE': 'ethwe', 'SHLVL': '1', 'AWS_REGION': 'us-west-1', 'NVIDIA_VISIBLE_DEVICES': 'all', 'TRAINING_JOB_NAME': 'pca-2019-03-08-00-07-38-756', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'ENVROOT': '/opt/amazon', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'OMP_NUM_THREADS': '2', 'HOSTNAME': 'aws', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/fd52a26a-7953-4a81-9ad0-e1f45c63a04b', 'PWD': '/', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2'}
[03/08/2019 00:10:18 INFO 140212411717440] envs={'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/2a343c78-c2c7-48e8-bf1d-79ad865e5656', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'DMLC_NUM_WORKER': '1', 'DMLC_PS_ROOT_PORT': '9000', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'SAGEMAKER_HTTP_PORT': '8080', 'HOME': '/root', 'PYTHONUNBUFFERED': 'TRUE', 'CANONICAL_ENVROOT': '/opt/amazon', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python2.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'LANG': 'en_US.utf8', 'DMLC_INTERFACE': 'ethwe', 'SHLVL': '1', 'DMLC_PS_ROOT_URI': '10.32.0.4', 'AWS_REGION': 'us-west-1', 'NVIDIA_VISIBLE_DEVICES': 'all', 'TRAINING_JOB_NAME': 'pca-2019-03-08-00-07-38-756', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'ENVROOT': '/opt/amazon', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'OMP_NUM_THREADS': '2', 'HOSTNAME': 'aws', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/fd52a26a-7953-4a81-9ad0-e1f45c63a04b', 'DMLC_ROLE': 'scheduler', 'PWD': '/', 'DMLC_NUM_SERVER': '1', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2'}
[03/08/2019 00:10:18 INFO 140212411717440] Launching parameter server for role server
[03/08/2019 00:10:18 INFO 140212411717440] {'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/2a343c78-c2c7-48e8-bf1d-79ad865e5656', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'SAGEMAKER_HTTP_PORT': '8080', 'HOME': '/root', 'PYTHONUNBUFFERED': 'TRUE', 'CANONICAL_ENVROOT': '/opt/amazon', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python2.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'LANG': 'en_US.utf8', 'DMLC_INTERFACE': 'ethwe', 'SHLVL': '1', 'AWS_REGION': 'us-west-1', 'NVIDIA_VISIBLE_DEVICES': 'all', 'TRAINING_JOB_NAME': 'pca-2019-03-08-00-07-38-756', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'ENVROOT': '/opt/amazon', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'OMP_NUM_THREADS': '2', 'HOSTNAME': 'aws', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/fd52a26a-7953-4a81-9ad0-e1f45c63a04b', 'PWD': '/', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2'}
[03/08/2019 00:10:18 INFO 140212411717440] envs={'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/2a343c78-c2c7-48e8-bf1d-79ad865e5656', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'DMLC_NUM_WORKER': '1', 'DMLC_PS_ROOT_PORT': '9000', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'SAGEMAKER_HTTP_PORT': '8080', 'HOME': '/root', 'PYTHONUNBUFFERED': 'TRUE', 'CANONICAL_ENVROOT': '/opt/amazon', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python2.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'LANG': 'en_US.utf8', 'DMLC_INTERFACE': 'ethwe', 'SHLVL': '1', 'DMLC_PS_ROOT_URI': '10.32.0.4', 'AWS_REGION': 'us-west-1', 'NVIDIA_VISIBLE_DEVICES': 'all', 'TRAINING_JOB_NAME': 'pca-2019-03-08-00-07-38-756', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'ENVROOT': '/opt/amazon', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'OMP_NUM_THREADS': '2', 'HOSTNAME': 'aws', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/fd52a26a-7953-4a81-9ad0-e1f45c63a04b', 'DMLC_ROLE': 'server', 'PWD': '/', 'DMLC_NUM_SERVER': '1', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2'}
[03/08/2019 00:10:18 INFO 140212411717440] Environment: {'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/2a343c78-c2c7-48e8-bf1d-79ad865e5656', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'DMLC_PS_ROOT_PORT': '9000', 'DMLC_NUM_WORKER': '1', 'SAGEMAKER_HTTP_PORT': '8080', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'PYTHONUNBUFFERED': 'TRUE', 'CANONICAL_ENVROOT': '/opt/amazon', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python2.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'LANG': 'en_US.utf8', 'DMLC_INTERFACE': 'ethwe', 'SHLVL': '1', 'DMLC_PS_ROOT_URI': '10.32.0.4', 'AWS_REGION': 'us-west-1', 'NVIDIA_VISIBLE_DEVICES': 'all', 'TRAINING_JOB_NAME': 'pca-2019-03-08-00-07-38-756', 'HOME': '/root', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'ENVROOT': '/opt/amazon', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'OMP_NUM_THREADS': '2', 'HOSTNAME': 'aws', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/fd52a26a-7953-4a81-9ad0-e1f45c63a04b', 'DMLC_ROLE': 'worker', 'PWD': '/', 'DMLC_NUM_SERVER': '1', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2'}
Process 64 is a shell:scheduler.
Process 73 is a shell:server.
Process 1 is a worker.
[03/08/2019 00:10:18 INFO 140212411717440] Using default worker.
[03/08/2019 00:10:18 INFO 140212411717440] Loaded iterator creator application/x-recordio-protobuf for content type ('application/x-recordio-protobuf', '1.0')
[03/08/2019 00:10:18 INFO 140212411717440] Loaded iterator creator application/x-labeled-vector-protobuf for content type ('application/x-labeled-vector-protobuf', '1.0')
[03/08/2019 00:10:18 INFO 140212411717440] Loaded iterator creator protobuf for content type ('protobuf', '1.0')
[2019-03-08 00:10:18.824] [tensorio] [info] batch={"data_pipeline": "/opt/ml/input/data/train", "num_examples": 500, "features": [{"name": "values", "shape": [34], "storage_type": "dense"}]}
[03/08/2019 00:10:18 INFO 140212411717440] Create Store: dist_sync
[03/08/2019 00:10:19 INFO 140212411717440] nvidia-smi took: 0.0251801013947 secs to identify 0 gpus
[03/08/2019 00:10:19 INFO 140212411717440] Number of GPUs being used: 0
[03/08/2019 00:10:19 INFO 140212411717440] The default executor is <PCAExecutor on cpu(0)>.
[03/08/2019 00:10:19 INFO 140212411717440] 34 feature(s) found in 'data'.
[03/08/2019 00:10:19 INFO 140212411717440] <PCAExecutor on cpu(0)> is assigned to batch slice from 0 to 499.
#metrics {"Metrics": {"initialize.time": {"count": 1, "max": 594.7449207305908, "sum": 594.7449207305908, "min": 594.7449207305908}}, "EndTime": 1552003819.421476, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "PCA"}, "StartTime": 1552003818.822317}

#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Number of Batches Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Number of Records Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Total Batches Seen": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Total Records Seen": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Max Records Seen Between Resets": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Reset Count": {"count": 1, "max": 0, "sum": 0.0, "min": 0}}, "EndTime": 1552003819.421682, "Dimensions": {"Host": "algo-1", "Meta": "init_train_data_iter", "Operation": "training", "Algorithm": "PCA"}, "StartTime": 1552003819.421629}

[2019-03-08 00:10:19.421] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 0, "duration": 597, "num_examples": 1}
#metrics {"Metrics": {"epochs": {"count": 1, "max": 1, "sum": 1.0, "min": 1}, "update.time": {"count": 1, "max": 36.4680290222168, "sum": 36.4680290222168, "min": 36.4680290222168}}, "EndTime": 1552003819.458467, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "PCA"}, "StartTime": 1552003819.42157}

[03/08/2019 00:10:19 INFO 140212411717440] #progress_metric: host=algo-1, completed 100 % of epochs
#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 7, "sum": 7.0, "min": 7}, "Number of Batches Since Last Reset": {"count": 1, "max": 7, "sum": 7.0, "min": 7}, "Number of Records Since Last Reset": {"count": 1, "max": 3218, "sum": 3218.0, "min": 3218}, "Total Batches Seen": {"count": 1, "max": 7, "sum": 7.0, "min": 7}, "Total Records Seen": {"count": 1, "max": 3218, "sum": 3218.0, "min": 3218}, "Max Records Seen Between Resets": {"count": 1, "max": 3218, "sum": 3218.0, "min": 3218}, "Reset Count": {"count": 1, "max": 1, "sum": 1.0, "min": 1}}, "EndTime": 1552003819.458807, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "PCA", "epoch": 0}, "StartTime": 1552003819.421963}

[03/08/2019 00:10:19 INFO 140212411717440] #throughput_metric: host=algo-1, train throughput=87029.1914449 records/second
#metrics {"Metrics": {"finalize.time": {"count": 1, "max": 19.690990447998047, "sum": 19.690990447998047, "min": 19.690990447998047}}, "EndTime": 1552003819.478824, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "PCA"}, "StartTime": 1552003819.458559}

[03/08/2019 00:10:19 INFO 140212411717440] Test data is not provided.
[2019-03-08 00:10:19.483] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 1, "duration": 54, "num_examples": 7}
[2019-03-08 00:10:19.483] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "duration": 651, "num_epochs": 2, "num_examples": 8}
#metrics {"Metrics": {"totaltime": {"count": 1, "max": 800.2028465270996, "sum": 800.2028465270996, "min": 800.2028465270996}, "setuptime": {"count": 1, "max": 41.87488555908203, "sum": 41.87488555908203, "min": 41.87488555908203}}, "EndTime": 1552003819.483391, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "PCA"}, "StartTime": 1552003819.478877}


2019-03-08 00:10:27 Training - Training image download completed. Training in progress.
2019-03-08 00:10:27 Uploading - Uploading generated training model
2019-03-08 00:10:27 Completed - Training job completed
Billable seconds: 23
CPU times: user 360 ms, sys: 18.5 ms, total: 379 ms
Wall time: 3min 11s
Accessing the PCA Model Attributes
After the model is trained, we can access the underlying model parameters.

Unzip the Model Details
Now that the training job is complete, you can find the job under Jobs in the Training subsection in the Amazon SageMaker console. You can find the job name listed in the training jobs. Use that job name in the following code to specify which model to examine.

Model artifacts are stored in S3 as a TAR file; a compressed file in the output path we specified + 'output/model.tar.gz'. The artifacts stored here can be used to deploy a trained model.

In [24]:
# Get the name of the training job, it's suggested that you copy-paste
# from the notebook or from a specific job in the AWS console
training_job_name='pca-2019-03-07-22-53-18-299'

# where the model is saved, by default
model_key = os.path.join(prefix, training_job_name, 'output/model.tar.gz')
print(model_key)

# download and unzip model
boto3.resource('s3').Bucket(bucket_name).download_file(model_key, 'model.tar.gz')

# unzipping as model_algo-1
os.system('tar -zxvf model.tar.gz')
os.system('unzip model_algo-1')
counties/pca-2019-03-07-22-53-18-299/output/model.tar.gz
Out[24]:
2304
MXNet Array
Many of the Amazon SageMaker algorithms use MXNet for computational speed, including PCA, and so the model artifacts are stored as an array. After the model is unzipped and decompressed, we can load the array using MXNet.

You can take a look at the MXNet documentation, here.

In [25]:
import mxnet as mx

# loading the unzipped artifacts
pca_model_params = mx.ndarray.load('model_algo-1')

# what are the params
print(pca_model_params)
{'s': 
[1.7896362e-02 3.0864021e-02 3.2130770e-02 3.5486195e-02 9.4831578e-02
 1.2699370e-01 4.0288666e-01 1.4084760e+00 1.5100485e+00 1.5957943e+00
 1.7783760e+00 2.1662524e+00 2.2966361e+00 2.3856051e+00 2.6954880e+00
 2.8067985e+00 3.0175958e+00 3.3952675e+00 3.5731301e+00 3.6966958e+00
 4.1890211e+00 4.3457499e+00 4.5410376e+00 5.0189657e+00 5.5786467e+00
 5.9809699e+00 6.3925138e+00 7.6952214e+00 7.9913125e+00 1.0180052e+01
 1.1718245e+01 1.3035975e+01 1.9592180e+01]
<NDArray 33 @cpu(0)>, 'v': 
[[ 2.46869749e-03  2.56468095e-02  2.50773830e-03 ... -7.63925165e-02
   1.59879066e-02  5.04589686e-03]
 [-2.80601848e-02 -6.86634064e-01 -1.96283013e-02 ... -7.59587288e-02
   1.57304872e-02  4.95312130e-03]
 [ 3.25766727e-02  7.17300594e-01  2.40726061e-02 ... -7.68136829e-02
   1.62378680e-02  5.13597298e-03]
 ...
 [ 1.12151138e-01 -1.17030945e-02 -2.88011521e-01 ...  1.39890045e-01
  -3.09406728e-01 -6.34506866e-02]
 [ 2.99992133e-02 -3.13433539e-03 -7.63589665e-02 ...  4.17341813e-02
  -7.06735924e-02 -1.42857227e-02]
 [ 7.33537527e-05  3.01008171e-04 -8.00925500e-06 ...  6.97060227e-02
   1.20169498e-01  2.33626723e-01]]
<NDArray 34x33 @cpu(0)>, 'mean': 
[[0.00988273 0.00986636 0.00989863 0.11017046 0.7560245  0.10094159
  0.0186819  0.02940491 0.0064698  0.01154038 0.31539047 0.1222766
  0.3030056  0.08220861 0.256217   0.2964254  0.28914267 0.40191284
  0.57868284 0.2854676  0.28294644 0.82774544 0.34378946 0.01576072
  0.04649627 0.04115358 0.12442778 0.47014    0.00980645 0.7608103
  0.19442631 0.21674445 0.0294168  0.22177474]]
<NDArray 1x34 @cpu(0)>}
PCA Model Attributes
Three types of model attributes are contained within the PCA model.

mean: The mean that was subtracted from a component in order to center it.
v: The makeup of the principal components; (same as ‘components_’ in an sklearn PCA model).
s: The singular values of the components for the PCA transformation. This does not exactly give the % variance from the original feature space, but can give the % variance from the projected feature space.
We are only interested in v and s.

From s, we can get an approximation of the data variance that is covered in the first n principal components. The approximate explained variance is given by the formula: the sum of squared s values for all top n components over the sum over squared s values for all components:

$$\begin{equation*}
\frac{\sum_{n}^{ } s_n^2}{\sum s^2}
\end{equation*}$$
From v, we can learn more about the combinations of original features that make up each principal component.

In [26]:
# get selected params
s=pd.DataFrame(pca_model_params['s'].asnumpy())
v=pd.DataFrame(pca_model_params['v'].asnumpy())
Data Variance
Our current PCA model creates 33 principal components, but when we create new dimensionality-reduced training data, we'll only select a few, top n components to use. To decide how many top components to include, it's helpful to look at how much data variance the components capture. For our original, high-dimensional data, 34 features captured 100% of our data variance. If we discard some of these higher dimensions, we will lower the amount of variance we can capture.

Tradeoff: dimensionality vs. data variance
As an illustrative example, say we have original data in three dimensions. So, three dimensions capture 100% of our data variance; these dimensions cover the entire spread of our data. The below images are taken from the PhD thesis, “Approaches to analyse and interpret biological profile data” by Matthias Scholz, (2006, University of Potsdam, Germany).

<img src='notebook_ims/3d_original_data.png' width=35% />

Now, you may also note that most of this data seems related; it falls close to a 2D plane, and just by looking at the spread of the data, we can visualize that the original, three dimensions have some correlation. So, we can instead choose to create two new dimensions, made up of linear combinations of the original, three dimensions. These dimensions are represented by the two axes/lines, centered in the data.

<img src='notebook_ims/pca_2d_dim_reduction.png' width=70% />

If we project this in a new, 2D space, we can see that we still capture most of the original data variance using just two dimensions. There is a tradeoff between the amount of variance we can capture and the number of component-dimensions we use to represent our data.

When we select the top n components to use in a new data model, we'll typically want to include enough components to capture about 80-90% of the original data variance. In this project, we are looking at generalizing over a lot of data and we'll aim for about 80% coverage.

Note: The top principal components, with the largest s values, are actually at the end of the s DataFrame. Let's print out the s values for the top n, principal components.

In [27]:
# looking at top 5 components
n_principal_components = 5

start_idx = N_COMPONENTS - n_principal_components  # 33-n

# print a selection of s
print(s.iloc[start_idx:, :])
            0
28   7.991313
29  10.180052
30  11.718245
31  13.035975
32  19.592180
EXERCISE: Calculate the explained variance
In creating new training data, you'll want to choose the top n principal components that account for at least 80% data variance.

Complete a function, explained_variance that takes in the entire array s and a number of top principal components to consider. Then return the approximate, explained variance for those top n components.

For example, to calculate the explained variance for the top 5 components, calculate s squared for each of the top 5 components, add those up and normalize by the sum of all squared s values, according to this formula:

$$\begin{equation*}
\frac{\sum_{5}^{ } s_n^2}{\sum s^2}
\end{equation*}$$
Using this function, you should be able to answer the question: What is the smallest number of principal components that captures at least 80% of the total variance in the dataset?

In [28]:
# Calculate the explained variance for the top n principal components
# you may assume you have access to the global var N_COMPONENTS
def explained_variance(s, n_top_components):
    '''Calculates the approx. data variance that n_top_components captures.
       :param s: A dataframe of singular values for top components; 
           the top value is in the last row.
       :param n_top_components: An integer, the number of top components to use.
       :return: The expected data variance covered by the n_top_components.'''
    
    start_idx = N_COMPONENTS - n_top_components  ## 33-3 = 30, for example
    # calculate approx variance
    exp_variance = np.square(s.iloc[start_idx:,:]).sum()/np.square(s).sum()
    
    return exp_variance[0]
Test Cell
Test out your own code by seeing how it responds to different inputs; does it return a reasonable value for the single, top component? What about for the top 5 components?

In [29]:
# test cell
n_top_components = 7 # select a value for the number of top components

# calculate the explained variance
exp_variance = explained_variance(s, n_top_components)
print('Explained variance: ', exp_variance)
Explained variance:  0.80167246
As an example, you should see that the top principal component accounts for about 32% of our data variance! Next, you may be wondering what makes up this (and other components); what linear combination of features make these components so influential in describing the spread of our data?

Below, let's take a look at our original features and use that as a reference.

In [30]:
# features
features_list = counties_scaled.columns.values
print('Features: \n', features_list)
Features: 
 ['TotalPop' 'Men' 'Women' 'Hispanic' 'White' 'Black' 'Native' 'Asian'
 'Pacific' 'Citizen' 'Income' 'IncomeErr' 'IncomePerCap' 'IncomePerCapErr'
 'Poverty' 'ChildPoverty' 'Professional' 'Service' 'Office' 'Construction'
 'Production' 'Drive' 'Carpool' 'Transit' 'Walk' 'OtherTransp'
 'WorkAtHome' 'MeanCommute' 'Employed' 'PrivateWork' 'PublicWork'
 'SelfEmployed' 'FamilyWork' 'Unemployment']
Component Makeup
We can now examine the makeup of each PCA component based on the weightings of the original features that are included in the component. The following code shows the feature-level makeup of the first component.

Note that the components are again ordered from smallest to largest and so I am getting the correct rows by calling N_COMPONENTS-1 to get the top, 1, component.

In [31]:
import seaborn as sns

def display_component(v, features_list, component_num, n_weights=10):
    
    # get index of component (last row - component_num)
    row_idx = N_COMPONENTS-component_num

    # get the list of weights from a row in v, dataframe
    v_1_row = v.iloc[:, row_idx]
    v_1 = np.squeeze(v_1_row.values)

    # match weights to features in counties_scaled dataframe, using list comporehension
    comps = pd.DataFrame(list(zip(v_1, features_list)), 
                         columns=['weights', 'features'])

    # we'll want to sort by the largest n_weights
    # weights can be neg/pos and we'll sort by magnitude
    comps['abs_weights']=comps['weights'].apply(lambda x: np.abs(x))
    sorted_weight_data = comps.sort_values('abs_weights', ascending=False).head(n_weights)

    # display using seaborn
    ax=plt.subplots(figsize=(10,6))
    ax=sns.barplot(data=sorted_weight_data, 
                   x="weights", 
                   y="features", 
                   palette="Blues_d")
    ax.set_title("PCA Component Makeup, Component #" + str(component_num))
    plt.show()
In [32]:
# display makeup of first component
num=2
display_component(v, counties_scaled.columns.values, component_num=num, n_weights=10)

Deploying the PCA Model
We can now deploy this model and use it to make "predictions". Instead of seeing what happens with some test data, we'll actually want to pass our training data into the deployed endpoint to create principal components for each data point.

Run the cell below to deploy/host this model on an instance_type that we specify.

In [33]:
%%time
# this takes a little while, around 7mins
pca_predictor = pca_SM.deploy(initial_instance_count=1, 
                              instance_type='ml.t2.medium')
INFO:sagemaker:Creating model with name: pca-2019-03-08-00-12-08-382
INFO:sagemaker:Creating endpoint with name pca-2019-03-08-00-07-38-756
--------------------------------------------------------------------------!CPU times: user 362 ms, sys: 36.5 ms, total: 398 ms
Wall time: 6min 15s
We can pass the original, numpy dataset to the model and transform the data using the model we created. Then we can take the largest n components to reduce the dimensionality of our data.

In [34]:
# pass np train data to the PCA model
train_pca = pca_predictor.predict(train_data_np)
In [35]:
# check out the first item in the produced training features
data_idx = 0
print(train_pca[data_idx])
label {
  key: "projection"
  value {
    float32_tensor {
      values: 0.0002009272575378418
      values: 0.0002455431967973709
      values: -0.0005782842636108398
      values: -0.0007815659046173096
      values: -0.00041911262087523937
      values: -0.0005133943632245064
      values: -0.0011316537857055664
      values: 0.0017268601804971695
      values: -0.005361668765544891
      values: -0.009066537022590637
      values: -0.008141040802001953
      values: -0.004735097289085388
      values: -0.00716288760304451
      values: 0.0003725700080394745
      values: -0.01208949089050293
      values: 0.02134685218334198
      values: 0.0009293854236602783
      values: 0.002417147159576416
      values: -0.0034637749195098877
      values: 0.01794189214706421
      values: -0.01639425754547119
      values: 0.06260128319263458
      values: 0.06637358665466309
      values: 0.002479255199432373
      values: 0.10011336207389832
      values: -0.1136140376329422
      values: 0.02589476853609085
      values: 0.04045158624649048
      values: -0.01082391943782568
      values: 0.1204797774553299
      values: -0.0883558839559555
      values: 0.16052711009979248
      values: -0.06027412414550781
    }
  }
}

EXERCISE: Create a transformed DataFrame
For each of our data points, get the top n component values from the list of component data points, returned by our predictor above, and put those into a new DataFrame.

You should end up with a DataFrame that looks something like the following:

                     c_1         c_2           c_3           c_4          c_5       ...
Alabama-Autauga    -0.060274    0.160527    -0.088356     0.120480    -0.010824    ...
Alabama-Baldwin    -0.149684    0.185969    -0.145743    -0.023092    -0.068677    ...
Alabama-Barbour    0.506202     0.296662     0.146258     0.297829    0.093111    ...
...
In [36]:
# create dimensionality-reduced data
def create_transformed_df(train_pca, counties_scaled, n_top_components):
    ''' Return a dataframe of data points with component features. 
        The dataframe should be indexed by State-County and contain component values.
        :param train_pca: A list of pca training data, returned by a PCA model.
        :param counties_scaled: A dataframe of normalized, original features.
        :param n_top_components: An integer, the number of top components to use.
        :return: A dataframe, indexed by State-County, with n_top_component values as columns.        
     '''
    # create new dataframe to add data to
    counties_transformed=pd.DataFrame()

    # for each of our new, transformed data points
    # append the component values to the dataframe
    for data in train_pca:
        # get component values for each data point
        components=data.label['projection'].float32_tensor.values
        counties_transformed=counties_transformed.append([list(components)])

    # index by county, just like counties_scaled
    counties_transformed.index=counties_scaled.index

    # keep only the top n components
    start_idx = N_COMPONENTS - n_top_components
    counties_transformed = counties_transformed.iloc[:,start_idx:]
    
    # reverse columns, component order     
    return counties_transformed.iloc[:, ::-1]
Now we can create a dataset where each county is described by the top n principle components that we analyzed earlier. Each of these components is a linear combination of the original feature space. We can interpret each of these components by analyzing the makeup of the component, shown previously.

In [38]:
# specify top n
top_n = 7

# call your function and create a new dataframe
counties_transformed = create_transformed_df(train_pca, counties_scaled, n_top_components=top_n)

# add descriptive columns
PCA_list=['c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6', 'c_7']
counties_transformed.columns=PCA_list 

# print result
counties_transformed.head()
Out[38]:
c_1	c_2	c_3	c_4	c_5	c_6	c_7
Alabama-Autauga	-0.060274	0.160527	-0.088356	0.120480	-0.010824	0.040452	0.025895
Alabama-Baldwin	-0.149684	0.185969	-0.145743	-0.023092	-0.068677	0.051573	0.048137
Alabama-Barbour	0.506202	0.296662	0.146258	0.297829	0.093111	-0.065244	0.107730
Alabama-Bibb	0.069224	0.190861	0.224402	0.011757	0.283526	0.017874	-0.092053
Alabama-Blount	-0.091030	0.254403	0.022714	-0.193824	0.100738	0.209945	-0.005099
Delete the Endpoint!
Now that we've deployed the model and created our new, transformed training data, we no longer need the PCA endpoint.

As a clean up step, you should always delete your endpoints after you are done using them (and if you do not plan to deploy them to a website, for example).

In [39]:
# delete predictor endpoint
session.delete_endpoint(pca_predictor.endpoint)
INFO:sagemaker:Deleting endpoint with name: pca-2019-03-08-00-07-38-756
Population Segmentation
Now, you’ll use the unsupervised clustering algorithm, k-means, to segment counties using their PCA attributes, which are in the transformed DataFrame we just created. K-means is a clustering algorithm that identifies clusters of similar data points based on their component makeup. Since we have ~3000 counties and 34 attributes in the original dataset, the large feature space may have made it difficult to cluster the counties effectively. Instead, we have reduced the feature space to 7 PCA components, and we’ll cluster on this transformed dataset.

EXERCISE: Define a k-means model
Your task will be to instantiate a k-means model. A KMeans estimator requires a number of parameters to be instantiated, which allow us to specify the type of training instance to use, and the model hyperparameters.

You can read about the required parameters, in the KMeans documentation; note that not all of the possible parameters are required.

Choosing a "Good" K
One method for choosing a "good" k, is to choose based on empirical data. A bad k would be one so high that only one or two very close data points are near it, and another bad k would be one so low that data points are really far away from the centers.

You want to select a k such that data points in a single cluster are close together but that there are enough clusters to effectively separate the data. You can approximate this separation by measuring how close your data points are to each cluster center; the average centroid distance between cluster points and a centroid. After trying several values for k, the centroid distance typically reaches some "elbow"; it stops decreasing at a sharp rate and this indicates a good value of k. The graph below indicates the average centroid distance for value of k between 5 and 12.

<img src='notebook_ims/elbow_graph.png' width=50% />

A distance elbow can be seen around 8 when the distance starts to increase and then decrease at a slower rate. This indicates that there is enough separation to distinguish the data points in each cluster, but also that you included enough clusters so that the data points aren’t extremely far away from each cluster.

In [40]:
# define a KMeans estimator
from sagemaker import KMeans

NUM_CLUSTERS = 8

kmeans = KMeans(role=role,
                train_instance_count=1,
                train_instance_type='ml.c4.xlarge',
                output_path=output_path, # using the same output path as was defined, earlier              
                k=NUM_CLUSTERS)
EXERCISE: Create formatted, k-means training data
Just as before, you should convert the counties_transformed df into a numpy array and then into a RecordSet. This is the required format for passing training data into a KMeans model.

In [41]:
# convert the transformed dataframe into record_set data
kmeans_train_data_np = counties_transformed.values.astype('float32')
kmeans_formatted_data = kmeans.record_set(kmeans_train_data_np)
EXERCISE: Train the k-means model
Pass in the formatted training data and train the k-means model.

In [42]:
%%time
# train kmeans
kmeans.fit(kmeans_formatted_data)
INFO:sagemaker:Creating training-job with name: kmeans-2019-03-08-00-37-22-788
2019-03-08 00:37:22 Starting - Starting the training job...
2019-03-08 00:37:27 Starting - Launching requested ML instances......
2019-03-08 00:38:31 Starting - Preparing the instances for training......
2019-03-08 00:39:53 Downloading - Downloading input data..
Docker entrypoint called with argument(s): train
[03/08/2019 00:40:07 INFO 140021904598848] Reading default configuration from /opt/amazon/lib/python2.7/site-packages/algorithm/resources/default-input.json: {u'_enable_profiler': u'false', u'_tuning_objective_metric': u'', u'_num_gpus': u'auto', u'local_lloyd_num_trials': u'auto', u'_log_level': u'info', u'_kvstore': u'auto', u'local_lloyd_init_method': u'kmeans++', u'force_dense': u'true', u'epochs': u'1', u'init_method': u'random', u'local_lloyd_tol': u'0.0001', u'local_lloyd_max_iter': u'300', u'_disable_wait_to_read': u'false', u'extra_center_factor': u'auto', u'eval_metrics': u'["msd"]', u'_num_kv_servers': u'1', u'mini_batch_size': u'5000', u'half_life_time_size': u'0', u'_num_slices': u'1'}
[03/08/2019 00:40:07 INFO 140021904598848] Reading provided configuration from /opt/ml/input/config/hyperparameters.json: {u'feature_dim': u'7', u'k': u'8', u'force_dense': u'True'}
[03/08/2019 00:40:07 INFO 140021904598848] Final configuration: {u'_tuning_objective_metric': u'', u'extra_center_factor': u'auto', u'local_lloyd_init_method': u'kmeans++', u'force_dense': u'True', u'epochs': u'1', u'feature_dim': u'7', u'local_lloyd_tol': u'0.0001', u'_disable_wait_to_read': u'false', u'eval_metrics': u'["msd"]', u'_num_kv_servers': u'1', u'mini_batch_size': u'5000', u'_enable_profiler': u'false', u'_num_gpus': u'auto', u'local_lloyd_num_trials': u'auto', u'_log_level': u'info', u'init_method': u'random', u'half_life_time_size': u'0', u'local_lloyd_max_iter': u'300', u'_kvstore': u'auto', u'k': u'8', u'_num_slices': u'1'}
[03/08/2019 00:40:07 WARNING 140021904598848] Loggers have already been setup.
Process 1 is a worker.
[03/08/2019 00:40:07 INFO 140021904598848] Using default worker.
[03/08/2019 00:40:07 INFO 140021904598848] Loaded iterator creator application/x-recordio-protobuf for content type ('application/x-recordio-protobuf', '1.0')
[03/08/2019 00:40:07 INFO 140021904598848] Create Store: local
[03/08/2019 00:40:07 INFO 140021904598848] nvidia-smi took: 0.0251669883728 secs to identify 0 gpus
[03/08/2019 00:40:07 INFO 140021904598848] Number of GPUs being used: 0
[2019-03-08 00:40:07.840] [tensorio] [info] batch={"data_pipeline": "/opt/ml/input/data/train", "num_examples": 5000, "features": [{"name": "values", "shape": [7], "storage_type": "dense"}]}
[03/08/2019 00:40:07 INFO 140021904598848] Setting up with params: {u'_tuning_objective_metric': u'', u'extra_center_factor': u'auto', u'local_lloyd_init_method': u'kmeans++', u'force_dense': u'True', u'epochs': u'1', u'feature_dim': u'7', u'local_lloyd_tol': u'0.0001', u'_disable_wait_to_read': u'false', u'eval_metrics': u'["msd"]', u'_num_kv_servers': u'1', u'mini_batch_size': u'5000', u'_enable_profiler': u'false', u'_num_gpus': u'auto', u'local_lloyd_num_trials': u'auto', u'_log_level': u'info', u'init_method': u'random', u'half_life_time_size': u'0', u'local_lloyd_max_iter': u'300', u'_kvstore': u'auto', u'k': u'8', u'_num_slices': u'1'}
/opt/amazon/lib/python2.7/site-packages/ai_algorithms_sdk/config/config_helper.py:172: DeprecationWarning: deprecated
  warnings.warn("deprecated", DeprecationWarning)
/opt/amazon/lib/python2.7/site-packages/ai_algorithms_sdk/config/config_helper.py:122: DeprecationWarning: deprecated
  warnings.warn("deprecated", DeprecationWarning)
[03/08/2019 00:40:07 INFO 140021904598848] Number of GPUs being used: 0
[03/08/2019 00:40:07 INFO 140021904598848] number of center slices 1
[03/08/2019 00:40:07 WARNING 140021904598848] Batch size 5000 is bigger than the first batch data. Effective batch size used to initialize is 3218
#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 1, "sum": 1.0, "min": 1}, "Number of Batches Since Last Reset": {"count": 1, "max": 1, "sum": 1.0, "min": 1}, "Number of Records Since Last Reset": {"count": 1, "max": 3218, "sum": 3218.0, "min": 3218}, "Total Batches Seen": {"count": 1, "max": 1, "sum": 1.0, "min": 1}, "Total Records Seen": {"count": 1, "max": 3218, "sum": 3218.0, "min": 3218}, "Max Records Seen Between Resets": {"count": 1, "max": 3218, "sum": 3218.0, "min": 3218}, "Reset Count": {"count": 1, "max": 0, "sum": 0.0, "min": 0}}, "EndTime": 1552005607.862528, "Dimensions": {"Host": "algo-1", "Meta": "init_train_data_iter", "Operation": "training", "Algorithm": "AWS/KMeansWebscale"}, "StartTime": 1552005607.862476}

[2019-03-08 00:40:07.862] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 0, "duration": 22, "num_examples": 1}
[03/08/2019 00:40:07 INFO 140021904598848] processed a total of 3218 examples
[03/08/2019 00:40:07 INFO 140021904598848] #progress_metric: host=algo-1, completed 100 % of epochs
#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 1, "sum": 1.0, "min": 1}, "Number of Batches Since Last Reset": {"count": 1, "max": 1, "sum": 1.0, "min": 1}, "Number of Records Since Last Reset": {"count": 1, "max": 3218, "sum": 3218.0, "min": 3218}, "Total Batches Seen": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Total Records Seen": {"count": 1, "max": 6436, "sum": 6436.0, "min": 6436}, "Max Records Seen Between Resets": {"count": 1, "max": 3218, "sum": 3218.0, "min": 3218}, "Reset Count": {"count": 1, "max": 1, "sum": 1.0, "min": 1}}, "EndTime": 1552005607.917481, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "AWS/KMeansWebscale", "epoch": 0}, "StartTime": 1552005607.862775}

[03/08/2019 00:40:07 INFO 140021904598848] #throughput_metric: host=algo-1, train throughput=58678.6812973 records/second
[03/08/2019 00:40:07 WARNING 140021904598848] wait_for_all_workers will not sync workers since the kv store is not running distributed
[03/08/2019 00:40:07 INFO 140021904598848] shrinking 80 centers into 8
[03/08/2019 00:40:07 INFO 140021904598848] local kmeans attempt #0. Current mean square distance 0.067297
[03/08/2019 00:40:07 INFO 140021904598848] local kmeans attempt #1. Current mean square distance 0.067115
[03/08/2019 00:40:08 INFO 140021904598848] local kmeans attempt #2. Current mean square distance 0.071168
[03/08/2019 00:40:08 INFO 140021904598848] local kmeans attempt #3. Current mean square distance 0.070390
[03/08/2019 00:40:08 INFO 140021904598848] local kmeans attempt #4. Current mean square distance 0.065784
[03/08/2019 00:40:08 INFO 140021904598848] local kmeans attempt #5. Current mean square distance 0.078768
[03/08/2019 00:40:08 INFO 140021904598848] local kmeans attempt #6. Current mean square distance 0.073432
[03/08/2019 00:40:08 INFO 140021904598848] local kmeans attempt #7. Current mean square distance 0.072345
[03/08/2019 00:40:08 INFO 140021904598848] local kmeans attempt #8. Current mean square distance 0.066644
[03/08/2019 00:40:08 INFO 140021904598848] local kmeans attempt #9. Current mean square distance 0.072777
[03/08/2019 00:40:08 INFO 140021904598848] finished shrinking process. Mean Square Distance = 0
[03/08/2019 00:40:08 INFO 140021904598848] #quality_metric: host=algo-1, train msd <loss>=0.0657843798399
[03/08/2019 00:40:08 INFO 140021904598848] compute all data-center distances: inner product took: 30.7196%, (0.017444 secs)
[03/08/2019 00:40:08 INFO 140021904598848] compute all data-center distances: point norm took: 24.3742%, (0.013841 secs)
[03/08/2019 00:40:08 INFO 140021904598848] batch data loading with context took: 11.3111%, (0.006423 secs)
[03/08/2019 00:40:08 INFO 140021904598848] predict compute msd took: 10.7598%, (0.006110 secs)
[03/08/2019 00:40:08 INFO 140021904598848] collect from kv store took: 5.3322%, (0.003028 secs)
[03/08/2019 00:40:08 INFO 140021904598848] splitting centers key-value pair took: 5.1387%, (0.002918 secs)
[03/08/2019 00:40:08 INFO 140021904598848] gradient: cluster center took: 4.2616%, (0.002420 secs)
[03/08/2019 00:40:08 INFO 140021904598848] gradient: one_hot took: 3.9022%, (0.002216 secs)
[03/08/2019 00:40:08 INFO 140021904598848] gradient: cluster size  took: 2.3260%, (0.001321 secs)
[03/08/2019 00:40:08 INFO 140021904598848] update state and report convergance took: 1.0669%, (0.000606 secs)
[03/08/2019 00:40:08 INFO 140021904598848] update set-up time took: 0.4383%, (0.000249 secs)
[03/08/2019 00:40:08 INFO 140021904598848] compute all data-center distances: center norm took: 0.3363%, (0.000191 secs)
[03/08/2019 00:40:08 INFO 140021904598848] predict minus dist took: 0.0332%, (0.000019 secs)
[03/08/2019 00:40:08 INFO 140021904598848] TOTAL took: 0.0567851066589
[03/08/2019 00:40:08 INFO 140021904598848] Number of GPUs being used: 0
#metrics {"Metrics": {"finalize.time": {"count": 1, "max": 260.5741024017334, "sum": 260.5741024017334, "min": 260.5741024017334}, "initialize.time": {"count": 1, "max": 16.234159469604492, "sum": 16.234159469604492, "min": 16.234159469604492}, "model.serialize.time": {"count": 1, "max": 0.17309188842773438, "sum": 0.17309188842773438, "min": 0.17309188842773438}, "update.time": {"count": 1, "max": 54.51202392578125, "sum": 54.51202392578125, "min": 54.51202392578125}, "epochs": {"count": 1, "max": 1, "sum": 1.0, "min": 1}, "state.serialize.time": {"count": 1, "max": 1.2581348419189453, "sum": 1.2581348419189453, "min": 1.2581348419189453}, "_shrink.time": {"count": 1, "max": 259.1099739074707, "sum": 259.1099739074707, "min": 259.1099739074707}}, "EndTime": 1552005608.179941, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/KMeansWebscale"}, "StartTime": 1552005607.838748}

[03/08/2019 00:40:08 INFO 140021904598848] Test data is not provided.
[2019-03-08 00:40:08.180] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "epoch": 1, "duration": 317, "num_examples": 1}
[2019-03-08 00:40:08.180] [tensorio] [info] data_pipeline_stats={"name": "/opt/ml/input/data/train", "duration": 339, "num_epochs": 2, "num_examples": 2}
#metrics {"Metrics": {"totaltime": {"count": 1, "max": 401.46708488464355, "sum": 401.46708488464355, "min": 401.46708488464355}, "setuptime": {"count": 1, "max": 12.835025787353516, "sum": 12.835025787353516, "min": 12.835025787353516}}, "EndTime": 1552005608.180324, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/KMeansWebscale"}, "StartTime": 1552005608.180065}


2019-03-08 00:40:18 Training - Training image download completed. Training in progress.
2019-03-08 00:40:18 Uploading - Uploading generated training model
2019-03-08 00:40:18 Completed - Training job completed
Billable seconds: 26
CPU times: user 362 ms, sys: 21.3 ms, total: 384 ms
Wall time: 3min 11s
EXERCISE: Deploy the k-means model
Deploy the trained model to create a kmeans_predictor.

In [43]:
%%time
# deploy the model to create a predictor
kmeans_predictor = kmeans.deploy(initial_instance_count=1, 
                                 instance_type='ml.t2.medium')
INFO:sagemaker:Creating model with name: kmeans-2019-03-08-00-40-34-502
INFO:sagemaker:Creating endpoint with name kmeans-2019-03-08-00-37-22-788
---------------------------------------------------------------------------------------!CPU times: user 436 ms, sys: 35.3 ms, total: 472 ms
Wall time: 7min 20s
EXERCISE: Pass in the training data and assign predicted cluster labels
After deploying the model, you can pass in the k-means training data, as a numpy array, and get resultant, predicted cluster labels for each data point.

In [44]:
# get the predicted clusters for all the kmeans training data
cluster_info=kmeans_predictor.predict(kmeans_train_data_np)
Exploring the resultant clusters
The resulting predictions should give you information about the cluster that each data point belongs to.

You should be able to answer the question: which cluster does a given data point belong to?

In [45]:
# print cluster info for first data point
data_idx = 0

print('County is: ', counties_transformed.index[data_idx])
print()
print(cluster_info[data_idx])
County is:  Alabama-Autauga

label {
  key: "closest_cluster"
  value {
    float32_tensor {
      values: 3.0
    }
  }
}
label {
  key: "distance_to_cluster"
  value {
    float32_tensor {
      values: 0.27046188712120056
    }
  }
}

Visualize the distribution of data over clusters
Get the cluster labels for each of our data points (counties) and visualize the distribution of points over each cluster.

In [46]:
# get all cluster labels
cluster_labels = [c.label['closest_cluster'].float32_tensor.values[0] for c in cluster_info]
In [47]:
# count up the points in each cluster
cluster_df = pd.DataFrame(cluster_labels)[0].value_counts()

print(cluster_df)
1.0    994
5.0    595
3.0    449
0.0    390
6.0    339
4.0    218
7.0    141
2.0     92
Name: 0, dtype: int64
In [49]:
# another method of visualizing the distribution
# display a histogram of cluster counts
ax =plt.subplots(figsize=(6,3))
ax = plt.hist(cluster_labels, bins=8,  range=(-0.5, 7.5), color='blue', rwidth=0.5)

title="Histogram of Cluster Counts"
plt.title(title, fontsize=12)
plt.show()

Now, you may be wondering, what do each of these clusters tell us about these data points? To improve explainability, we need to access the underlying model to get the cluster centers. These centers will help describe which features characterize each cluster.

Delete the Endpoint!
Now that you've deployed the k-means model and extracted the cluster labels for each data point, you no longer need the k-means endpoint.

In [50]:
# delete kmeans endpoint
session.delete_endpoint(kmeans_predictor.endpoint)
INFO:sagemaker:Deleting endpoint with name: kmeans-2019-03-08-00-37-22-788
Model Attributes & Explainability
Explaining the result of the modeling is an important step in making use of our analysis. By combining PCA and k-means, and the information contained in the model attributes within a SageMaker trained model, you can learn about a population and remark on some patterns you've found, based on the data.

EXERCISE: Access the k-means model attributes
Extract the k-means model attributes from where they are saved as a TAR file in an S3 bucket.

You'll need to access the model by the k-means training job name, and then unzip the file into model_algo-1. Then you can load that file using MXNet, as before.

In [51]:
# download and unzip the kmeans model file
kmeans_job_name = 'kmeans-2019-03-08-00-37-22-788'

model_key = os.path.join(prefix, kmeans_job_name, 'output/model.tar.gz')

# download the model file
boto3.resource('s3').Bucket(bucket_name).download_file(model_key, 'model.tar.gz')
os.system('tar -zxvf model.tar.gz')
os.system('unzip model_algo-1')
Out[51]:
2304
In [52]:
# get the trained kmeans params using mxnet
kmeans_model_params = mx.ndarray.load('model_algo-1')

print(kmeans_model_params)
[
[[ 3.68001163e-01  2.26354256e-01  7.88291916e-02  2.61262119e-01
   9.61402357e-02 -5.05092032e-02  3.57557014e-02]
 [-7.32245147e-02  1.05659112e-01  1.13722928e-01 -6.33113831e-02
  -3.95483039e-02  4.66217995e-02 -2.19286084e-02]
 [ 1.31537044e+00 -2.57377833e-01 -1.57875896e-01 -4.15273398e-01
  -1.25370100e-01  1.16665870e-01  1.62481740e-01]
 [-1.62926733e-01  4.53134663e-02 -3.00800741e-01  7.25739002e-02
  -4.14044000e-02  5.90644330e-02  8.24726187e-04]
 [ 3.09595257e-01 -1.57976970e-01 -1.62361398e-01 -1.85713798e-01
   1.38460830e-01 -1.14379212e-01 -6.49842843e-02]
 [-2.69347250e-01  8.37490428e-03 -1.81501284e-02 -7.30615258e-02
   4.57714126e-03 -8.09832960e-02 -1.25052175e-02]
 [-2.39941925e-01 -3.17158133e-01  1.06285624e-01  4.89501953e-02
   4.87177409e-02 -6.73310785e-03  1.10704802e-01]
 [ 2.05574751e-01 -4.47796524e-01  8.54142755e-02  1.98281378e-01
  -1.12922713e-01 -1.60498768e-02 -1.51252389e-01]]
<NDArray 8x7 @cpu(0)>]
There is only 1 set of model parameters contained within the k-means model: the cluster centroid locations in PCA-transformed, component space.

centroids: The location of the centers of each cluster in component space, identified by the k-means algorithm.
In [53]:
# get all the centroids
cluster_centroids=pd.DataFrame(kmeans_model_params[0].asnumpy())
cluster_centroids.columns=counties_transformed.columns

display(cluster_centroids)
c_1	c_2	c_3	c_4	c_5	c_6	c_7
0	0.368001	0.226354	0.078829	0.261262	0.096140	-0.050509	0.035756
1	-0.073225	0.105659	0.113723	-0.063311	-0.039548	0.046622	-0.021929
2	1.315370	-0.257378	-0.157876	-0.415273	-0.125370	0.116666	0.162482
3	-0.162927	0.045313	-0.300801	0.072574	-0.041404	0.059064	0.000825
4	0.309595	-0.157977	-0.162361	-0.185714	0.138461	-0.114379	-0.064984
5	-0.269347	0.008375	-0.018150	-0.073062	0.004577	-0.080983	-0.012505
6	-0.239942	-0.317158	0.106286	0.048950	0.048718	-0.006733	0.110705
7	0.205575	-0.447797	0.085414	0.198281	-0.112923	-0.016050	-0.151252
Visualizing Centroids in Component Space
You can't visualize 7-dimensional centroids in space, but you can plot a heatmap of the centroids and their location in the transformed feature space.

This gives you insight into what characteristics define each cluster. Often with unsupervised learning, results are hard to interpret. This is one way to make use of the results of PCA + clustering techniques, together. Since you were able to examine the makeup of each PCA component, you can understand what each centroid represents in terms of the PCA components.

In [54]:
# generate a heatmap in component space, using the seaborn library
plt.figure(figsize = (12,9))
ax = sns.heatmap(cluster_centroids.T, cmap = 'YlGnBu')
ax.set_xlabel("Cluster")
plt.yticks(fontsize = 16)
plt.xticks(fontsize = 16)
ax.set_title("Attribute Value by Centroid")
plt.show()

If you've forgotten what each component corresponds to at an original-feature-level, that's okay! You can use the previously defined display_component function to see the feature-level makeup.

In [56]:
# what do each of these components mean again?
# let's use the display function, from above
component_num=4
display_component(v, counties_scaled.columns.values, component_num=component_num)

Natural Groupings
You can also map the cluster labels back to each individual county and examine which counties are naturally grouped together.

In [57]:
# add a 'labels' column to the dataframe
counties_transformed['labels']=list(map(int, cluster_labels))

# sort by cluster label 0-6
sorted_counties = counties_transformed.sort_values('labels', ascending=True)
# view some pts in cluster 0
sorted_counties.head(20)
Out[57]:
c_1	c_2	c_3	c_4	c_5	c_6	c_7	labels
Georgia-Candler	0.363197	0.109423	0.204951	0.059516	0.067597	-0.019374	0.030013	0
Louisiana-Concordia	0.409779	0.204391	0.191335	0.268234	0.007950	-0.054527	0.181117	0
Virginia-Newport News city	0.261421	0.177800	-0.229566	0.321782	0.014855	-0.102143	-0.024013	0
Louisiana-De Soto	0.295931	0.348835	0.048717	0.164811	0.166941	0.010911	0.040519	0
Florida-Gadsden	0.662961	0.165212	0.012804	0.476595	0.042991	0.093047	0.007101	0
Virginia-Norfolk city	0.345661	0.123631	-0.143410	0.363603	-0.007470	-0.083678	-0.005115	0
Louisiana-East Baton Rouge	0.242962	0.229884	-0.259422	0.379246	0.018616	-0.107516	0.071976	0
Virginia-Petersburg city	0.792302	0.426670	-0.051610	0.611594	0.033392	-0.232590	0.001919	0
Louisiana-East Carroll	0.933136	0.293730	0.356297	0.540606	-0.054206	-0.278123	0.160721	0
Louisiana-East Feliciana	0.286842	0.201158	0.002600	0.365350	0.140875	0.131723	0.048354	0
Virginia-Portsmouth city	0.400933	0.202716	-0.105670	0.452546	0.072539	-0.063671	0.036404	0
Louisiana-Evangeline	0.232242	0.243974	0.159740	0.076092	0.137570	0.109930	0.080670	0
Louisiana-Franklin	0.303057	0.121336	0.236197	0.172664	0.046740	-0.030970	0.258763	0
Virginia-Richmond city	0.407804	0.227938	-0.214292	0.420849	-0.057985	-0.131112	0.041372	0
Louisiana-Iberia	0.169558	0.269125	-0.033261	0.117168	0.123274	-0.094210	0.044880	0
Louisiana-Iberville	0.289200	0.275354	-0.043679	0.319326	0.176963	-0.114928	-0.003577	0
Virginia-Roanoke city	0.183414	0.235204	-0.076873	0.146362	-0.108143	-0.183366	-0.000494	0
Louisiana-Jackson	0.225916	0.198804	0.142211	0.151452	0.143448	0.102662	0.047744	0
North Carolina-Perquimans	0.127545	-0.034301	0.110735	0.231734	0.161234	0.260116	0.119911	0
Mississippi-Jasper	0.384125	0.423346	0.102244	0.286187	0.282340	-0.135209	0.000461	0
You can also examine one of the clusters in more detail, like cluster 1, for example. A quick glance at the location of the centroid in component space (the heatmap) tells us that it has the highest value for the comp_6 attribute. You can now see which counties fit that description.

In [58]:
# get all counties with label == 1
cluster=counties_transformed[counties_transformed['labels']==1]
cluster.head()
Out[58]:
c_1	c_2	c_3	c_4	c_5	c_6	c_7	labels
Alabama-Bibb	0.069224	0.190861	0.224402	0.011757	0.283526	0.017874	-0.092053	1
Alabama-Blount	-0.091030	0.254403	0.022714	-0.193824	0.100738	0.209945	-0.005099	1
Alabama-Calhoun	0.128913	0.223409	0.070180	0.081091	-0.069090	0.002235	0.012866	1
Alabama-Cherokee	-0.080311	0.104444	0.210828	-0.061823	0.027139	0.107847	-0.002206	1
Alabama-Chilton	0.022630	0.240691	0.068429	-0.103816	0.138959	0.141059	-0.052229	1
Final Cleanup!
Double check that you have deleted all your endpoints.
I'd also suggest manually deleting your S3 bucket, models, and endpoint configurations directly from your AWS console.
You can find thorough cleanup instructions, in the documentation.

Conclusion
You have just walked through a machine learning workflow for unsupervised learning, specifically, for clustering a dataset using k-means after reducing the dimensionality using PCA. By accessing the underlying models created within SageMaker, you were able to improve the explainability of your model and draw insights from the resultant clusters.

Using these techniques, you have been able to better understand the essential characteristics of different counties in the US and segment them into similar groups, accordingly.
