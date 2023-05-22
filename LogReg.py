#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("D:\Python\insurance_data.csv")
df


# ### Plot a Scatter plot of data

# In[4]:


plt.scatter(df.age, df.bought_insurance, marker='+',color='red')


# In[5]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df[['age']],df.bought_insurance,test_size=0.2)
X_train


# In[6]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)


# In[7]:


model.predict(X_test)


# In[8]:


y_test


# In[9]:


model.score(X_test,y_test)


# ### probabilities of prediction

# In[12]:


model.predict_proba(X_test)


# In[ ]:




