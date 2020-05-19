# -*- coding: utf-8 -*-
"""
Created on Sat May  9 08:16:42 2020

@author: opawar
"""

#importing the necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
%matplotlib qt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

#reading the datasets into dataframes

df_train = pd.read_csv("C:/Users/opawar/Desktop/Python/Projects/Titanic/train.csv")

#mapping Cabin to Cabin Letter values to categories
def cabin_letter(x):
    if(x=='nan'):
        return 'o'
    else:
        y=str(x)[:1]
        return y
df_train['Cabin'] = df_train['Cabin'].apply(lambda x: cabin_letter(x))

#mapping Ticket to Digit or Alpha values to categories
import re
def Ticket_letter(x):
    if(x.isdigit()):
        return 'Ticket'
    else:
        x=x.split(' ')[0]
        x=re.sub('[^A-Za-z0-9]+', '', x)
        return x
df_train['Ticket_Section'] = df_train['Ticket'].apply(lambda x: Ticket_letter(x))

#mapping Name values to categories
def Honorific(x):
        x=x.split('.')[0]
        x=x.split(' ')[-1]
        return x
df_train['Honorific'] = df_train['Name'].apply(lambda x: Honorific(x))

#imputing median for missing 'Age' values
med_imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
df_train['Age'] = med_imputer.fit_transform(df_train[['Age']])

#mapping age values to categories
def age_map(x):
    if(x <= 12):
        return "Kid"
    elif(x >= 13 and x <= 19):
        return "Teen"
    elif(x > 19 and x < 70):
        return "Adult"
    else:
        return "Senior"
df_train['Age'] = df_train['Age'].apply(lambda x: age_map(x))

#processing 'Embarked' columns
df_train['Embarked'].fillna('S', inplace=True)

#converting 'Fare' values to categorical
df_train['Fare'] = pd.cut(df_train['Fare'], bins=[-1, 7, 11, 15, 22, 40, 520], labels=[1, 2, 3, 4, 5, 6])

#Dropping Columns
df_train.drop(['Name','Ticket','PassengerId'], axis=1, inplace=True)

df_train.isnull().values.any()
df_train.isnull().sum()*100/len(df_train)

#Display Stats
fig, ax = plt.subplots(1,2,figsize=(10,5))
sns.countplot(df_train['Age'], data=df_train, ax=ax[0])
sns.countplot(df_train['Age'], hue='Survived', data=df_train, ax=ax[1])
ax[0].title.set_text("Frequency of each age group")
ax[1].title.set_text("Survived: Age Group")

fig, ax = plt.subplots(1,2, figsize=(10,5))
sns.countplot(df_train['Sex'], data=df_train, ax=ax[0])
sns.countplot(df_train['Sex'], hue='Survived', data=df_train, ax=ax[1])
ax[0].title.set_text("Frequency: Sex")
ax[1].title.set_text("Survived: Sex")

fig, ax = plt.subplots(1,2, figsize=(10,5))
sns.countplot(df_train['Pclass'], data=df_train, ax=ax[0])
sns.countplot(df_train['Pclass'], hue='Survived', data=df_train, ax=ax[1])
ax[0].title.set_text("Frequency: Pclass")
ax[1].title.set_text("Survived: Pclass")

fig, ax = plt.subplots(1,2, figsize=(10,5))
sns.countplot(df_train['SibSp'], data=df_train, ax=ax[0])
sns.countplot(df_train['SibSp'], hue='Survived', data=df_train, ax=ax[1])
ax[0].title.set_text("Frequency: SibSp")
ax[1].title.set_text("Survived: SibSp")

fig, ax = plt.subplots(1,2, figsize=(10,5))
sns.countplot(df_train['Embarked'], data=df_train, ax=ax[0])
sns.countplot(df_train['Embarked'], hue='Survived', data=df_train, ax=ax[1])
ax[0].title.set_text("Frequency: Embarked")
ax[1].title.set_text("Survived: Embarked")

fig, ax = plt.subplots(1,2, figsize=(10,5))
sns.countplot(df_train['Parch'], data=df_train, ax=ax[0])
sns.countplot(df_train['Parch'], hue='Survived', data=df_train, ax=ax[1])
ax[0].title.set_text("Frequency: Parch")
ax[1].title.set_text("Survived: Parch")

fig, ax = plt.subplots(1,2, figsize=(10,5))
sns.countplot(df_train['Cabin'], data=df_train, ax=ax[0])
sns.countplot(df_train['Cabin'], hue='Survived', data=df_train, ax=ax[1])
ax[0].title.set_text("Frequency: Cabin")
ax[1].title.set_text("Survived: Cabin")

fig, ax = plt.subplots(1,2, figsize=(10,5))
sns.countplot(df_train['Ticket_Section'], data=df_train, ax=ax[0])
sns.countplot(df_train['Ticket_Section'], hue='Survived', data=df_train, ax=ax[1])
ax[0].title.set_text("Frequency: Ticket Section")
ax[1].title.set_text("Survived: Ticket Section")

fig, ax = plt.subplots(1,2, figsize=(10,5))
sns.countplot(df_train['Honorific'], data=df_train, ax=ax[0])
sns.countplot(df_train['Honorific'], hue='Survived', data=df_train, ax=ax[1])
ax[0].title.set_text("Frequency: Honorific")
ax[1].title.set_text("Survived: Honorific")

#plotting a heatmap of the train set
plt.figure(figsize=(10,10))
sns.heatmap(df_train.corr(), xticklabels = df_train.columns.values, yticklabels = df_train.columns.values, annot=True, cmap="YlGnBu")

df_train.head(10)

#LabelEncoder
LESex = LabelEncoder()
LEEmbarked = LabelEncoder()
LEAge = LabelEncoder()
LECabin = LabelEncoder()
LEHonorific = LabelEncoder()
LETicket_Section = LabelEncoder()
#label encoding the remaining categorical and continous variables
df_train['Sex'] = LESex.fit_transform(df_train['Sex'])
np.save('LESex.npy', LESex.classes_)
df_train['Embarked'] = LEEmbarked.fit_transform(df_train['Embarked'])
np.save('LEEmbarked.npy', LEEmbarked.classes_)
df_train['Age'] = LEAge.fit_transform(df_train['Age'])
np.save('LEAge.npy', LEAge.classes_)
df_train['Cabin'] = LECabin.fit_transform(df_train['Cabin'])
np.save('LECabin.npy', LECabin.classes_)
df_train['Honorific'] = LEHonorific.fit_transform(df_train['Honorific'])
np.save('LEHonorific.npy', LEHonorific.classes_)
df_train['Ticket_Section'] = LETicket_Section.fit_transform(df_train['Ticket_Section'])
np.save('LETicket_Section.npy', LETicket_Section.classes_)

#sorting PassendgerId in ascending order
#df_train.sort_values(by=['PassengerId'], inplace=True)

#Splitting the train set into dependent and independent variables
y = df_train['Survived']
X = df_train.drop('Survived', axis = 1)

#converting 'Fare' values to int64 type
X['Fare'] = X['Fare'].astype('int64')

#train-test split
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.65, test_size = 0.35, random_state=100 )

#creating a RandomForestClassifier object and generate the model
model = RandomForestClassifier(n_estimators=100)

# Fit on training data
model.fit(X, y)

#Save pickle
import pickle
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

#Predict Values
#y_pred=model.predict(X_valid)

#confusion metrix
#from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(y_valid, y_pred)

#from sklearn.metrics import classification_report 
#cr=classification_report(y_valid, y_pred) 

#-----------------------------------------------------------------------


df_test = pd.read_csv("C:/Users/opawar/Desktop/Python/Projects/Titanic/test.csv")


#mapping Cabin to Cabin Letter values to categories
def cabin_letter(x):
    if(x=='nan'):
        return 'o'
    else:
        y=str(x)[:1]
        return y
df_test['Cabin'] = df_test['Cabin'].apply(lambda x: cabin_letter(x))

#mapping Ticket to Digit or Alpha values to categories
import re
def Ticket_letter(x):
    if(x.isdigit()):
        return 'Ticket'
    else:
        x=x.split(' ')[0]
        x=re.sub('[^A-Za-z0-9]+', '', x)
        return x
df_test['Ticket_Section'] = df_test['Ticket'].apply(lambda x: Ticket_letter(x))

#mapping Name values to categories
def Honorific(x):
        x=x.split('.')[0]
        x=x.split(' ')[-1]
        return x
df_test['Honorific'] = df_test['Name'].apply(lambda x: Honorific(x))

#imputing median for missing 'Age' values
med_imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
df_test['Age'] = med_imputer.fit_transform(df_test[['Age']])

#mapping age values to categories
def age_map(x):
    if(x <= 12):
        return "Kid"
    elif(x >= 13 and x <= 19):
        return "Teen"
    elif(x > 19 and x < 70):
        return "Adult"
    else:
        return "Senior"
df_test['Age'] = df_test['Age'].apply(lambda x: age_map(x))

#processing 'Embarked' columns
df_test['Embarked'].fillna('S', inplace=True)

#converting 'Fare' values to categorical
df_test['Fare'] = pd.cut(df_test['Fare'], bins=[-1, 7, 11, 15, 22, 40, 520], labels=[1, 2, 3, 4, 5, 6])

#Dropping Columns
df_test.drop(['Name','Ticket','PassengerId'], axis=1, inplace=True)

#Transform the Data Model:
        LESex = LabelEncoder()
        LESex.classes_ = np.load('LESex.npy',allow_pickle=True)
        LEEmbarked = LabelEncoder()
        LEEmbarked.classes_ = np.load('LEEmbarked.npy',allow_pickle=True)
        LEAge = LabelEncoder()
        LEAge.classes_ = np.load('LEAge.npy',allow_pickle=True)
        LECabin = LabelEncoder()
        LECabin.classes_ = np.load('LECabin.npy',allow_pickle=True)
        LEHonorific = LabelEncoder()
        LEHonorific.classes_ = np.load('LEHonorific.npy',allow_pickle=True)
        LETicket_Section = LabelEncoder()
        LETicket_Section.classes_ = np.load('LETicket_Section.npy',allow_pickle=True)
        df_test['Sex']=LESex.transform(df_test['Sex'])
        df_test['Age']=LEAge.transform(df_test['Age'])
        df_test['Cabin']=LECabin.transform(df_test['Cabin'])
        df_test['Embarked']=LEEmbarked.transform(df_test['Embarked'])
        df_test['Ticket_Section']=LETicket_Section.transform(df_test['Ticket_Section'])
        df_test['Honorific']=LEHonorific.transform(df_test['Honorific'])
        
#sorting PassendgerId in ascending order
df_test.sort_values(by=['PassengerId'], inplace=True)

#converting 'Fare' values to int64 type
X['Fare'] = X['Fare'].astype('int64')

df_test.isnull().values.any()
df_test.isnull().sum()*100/len(df_train)


#making predictions on the test set
y_test_pred = model.predict(df_test)


