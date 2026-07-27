<H3>NAME: LAKSHANYA.N </H3>
<H3>REGISTER NO: 212224230136</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 27/027/2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
#import libraries
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```
```
#Read the dataset from drive
df = pd.read_csv('student_performance_dataset.csv')
print(df)
#split the dataset
X = df.iloc[:, :-1].values
print(X)
y = df.iloc[:, -1].values
print(y)
# Finding Missing Values
print(df.isnull().sum())
#Handling Missing values
df.fillna(df.mean(numeric_only=True).round(1), inplace=True)
df['parental_education'].fillna(df['parental_education'].mode()[0], inplace=True)
print(df.isnull().sum())
y = df.iloc[:, -1].values
print(y)
```
```
#Check for Duplicates
df.duplicated()
#Detect Outliers
print(df['final_exam_score'].describe())
#When we normalize the dataset it brings the value of all the features
#between 0 and 1 so that all the columns are in the same range,
# and thus there is no dominant feature.
numeric_cols = df.select_dtypes(include='number').columns
scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
print(df1)
```
```
#splitting the data for training & Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#'test_size=0.2' means 20% test data and 80% train data
print(X_train)
print(len(X_train))
print(X_test)
print(len(X_test))

print(y_train)
print(len(y_train))
print(y_test)
print(len(y_test))
```
## OUTPUT:
<img width="666" height="707" alt="image" src="https://github.com/user-attachments/assets/c52929fa-2602-4594-b631-51ec3299e8ad" />

<img width="562" height="696" alt="image" src="https://github.com/user-attachments/assets/913dc0aa-f6ae-48d5-9c23-7fa08e3b3b2a" />

<img width="620" height="701" alt="image" src="https://github.com/user-attachments/assets/449ef11e-8fae-4775-89de-267a62f73772" />

<img width="702" height="592" alt="image" src="https://github.com/user-attachments/assets/b8c92aec-7633-4354-8a9f-42eb37656748" />

<img width="653" height="635" alt="image" src="https://github.com/user-attachments/assets/b3241853-6deb-4d16-b384-120ed09a68c9" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


