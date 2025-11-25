#!/usr/bin/env python
# coding: utf-8

# In[33]:


# import libraries:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



# In[2]:


#Loading the dataset to a Pandas Dataframe

credit_card_data = pd.read_csv('creditcard.csv')


# In[3]:


# let's see first 5 rows of the dataset:

credit_card_data.head(5)


# In[4]:


# let's see last 5 rows of our dataset:

credit_card_data.tail()


# In[5]:


# dataset information:

credit_card_data.info()


# In[6]:


# checking number of missing values:

credit_card_data.isnull().sum()


# In[7]:


# Find distribution of Normal transaction or Fraud transaction:

credit_card_data['Class'].value_counts()


# In[8]:


# Seperating the data:

normal = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[9]:


print(normal.shape)
print(fraud.shape)


# In[10]:


# statistical measures of the data:

normal.Amount.describe()


# In[11]:


fraud.Amount.describe()


# In[12]:


# visualize the data:

sns.relplot(x = 'Amount' , y = 'Time' , hue = 'Class', data = credit_card_data)


# In[13]:


# Compare values of both transactions:

credit_card_data.groupby('Class').mean()


# In[14]:


# Now we will build a sample dataset containing similar distribution of normal transaction and fraud transaction:

normal_sample = normal.sample(n=492)


# In[15]:


# Concate two dataframes:

credit_card_new_data = pd.concat([normal_sample, fraud], axis=0)


# In[16]:


credit_card_new_data


# In[17]:


credit_card_new_data['Class'].value_counts()


# In[18]:


# Splitting data into features and targets

X = credit_card_new_data.drop('Class', axis=1)
Y = credit_card_new_data['Class']


# In[19]:


X


# In[20]:


Y


# In[26]:


# splitting the data into training and testing data:

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify = Y, random_state= 2)


# In[27]:


print(X.shape, X_train.shape,  X_test.shape)


# In[28]:


# Creating Model:

model = LogisticRegression()

# training the Logistic Regression model with training data:

model.fit(X_train,Y_train)


# In[30]:


# Model Evaluation

X_train_pred = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_pred, Y_train)

print('Accuracy of Training data:', training_data_accuracy)


# In[31]:


# accuracy on test data:

X_test_pred = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_pred, Y_test)

print('Accuracy of Testing data:', test_data_accuracy)


# In[32]:


print(confusion_matrix(X_test_pred, Y_test))


# In[34]:


print(classification_report(X_test_pred, Y_test))


# In[35]:


print(classification_report(X_train_pred, Y_train))


# In[ ]:




