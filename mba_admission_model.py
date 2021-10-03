#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import flask
'''
https://www.kaggle.com/nitindatta/graduate-admission-chances with changes
'''


# In[2]:


df = pd.read_csv("Admission_Predict.csv")
df.head(10)


# In[3]:


df.describe()


# In[4]:


df.isnull().sum()


# In[5]:


df = df.rename(columns = {'Chance of Admit ':'Chance of Admit'})
X = df.drop(['Serial No.','Chance of Admit'],axis=1)
y = df['Chance of Admit']


# In[6]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=101)


# In[7]:


'''
https://pub.towardsai.net/machine-learning-algorithms-for-beginners-with-python-code-examples-ml-19c6afd60daa
'''

from sklearn import linear_model

model = linear_model.LinearRegression()
model.fit(X_train,y_train)

coefficient = list(model.coef_)
intercept = model.intercept_
print("coefficients:", coefficient)
print("intercept:", intercept)


# In[16]:


#Evaluate model

y_pred = model.predict(X_test)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[9]:


#Predicting future values
xnew = (np.array([340, 110, 5, 5, 4, 9.1, 0])).reshape(1,-1)
ynew = model.predict(xnew)
print('chance of admission:', float(ynew))


# In[11]:


#Save the model to pickle
import pickle

with open('model_pickle', 'wb') as f:
    pickle.dump(model, f)


# In[ ]:




