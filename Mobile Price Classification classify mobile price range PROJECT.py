#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df_t = pd.read_csv(r'C:\Users\shiva\Desktop\dataset\train.csv')
df_te =  pd.read_csv(r'C:\Users\shiva\Desktop\dataset\test.csv')


# In[3]:


df_t


# In[4]:


df_te


# In[5]:


import seaborn as sns


# In[5]:


import matplotlib.pyplot as plt 


# In[6]:


matplotlib inline


# In[7]:


df_t.shape


# In[8]:


df_t.describe()


# In[9]:


df_t.info()


# In[10]:


plt.figure(figsize = (12,6))
sns.heatmap(df_t.corr())
plt.show()


# In[11]:


#Plotting Relation between Price Range & Battery Power
plt.figure(figsize = (12,6))
sns.barplot(x = 'price_range' , y = 'battery_power' , data = df_t)
plt.show()


# In[12]:


#Plotting Relation Between Price Range & pixel Height / Width
plt.figure(figsize = (14,6))
plt.subplot(1,2,1)
sns.barplot(x = 'price_range' , y = 'px_height' , data = df_t , palette = 'Oranges')
plt.subplot(1,2,2)
sns.barplot(x = 'price_range' , y = 'px_width' , data = df_t , palette = 'Greens')
plt.show()


# In[13]:


#Plotting Relation  between Price Range & RAM
plt.figure(figsize = (12,6))
sns.barplot(x = 'price_range' , y = 'ram' , data = df_t)
plt.show()


# In[14]:


#Plotting Relation Between Price Range & 3G/4G
plt.figure(figsize = (12,6))
sns.countplot(df_t['three_g'] , hue = df_t['price_range'] , palette = 'pink')
plt.show()


# In[15]:


plt.figure(figsize = (12,6))
sns.countplot(df_t['four_g'] , hue = df_t['price_range'] , palette = 'ocean')
plt.show()


# In[16]:


#Plotting Relation between Price Range & Memory
plt.figure(figsize = (12,6))
sns.lineplot(x = 'price_range' , y = 'int_memory' , data = df_t , hue = 'dual_sim')
plt.show()


# In[17]:


#Data Preprocessing
x = df_t.drop(['price_range'] , axis = 1)
y = df_t['price_range']


# In[18]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train , y_test = train_test_split(x,y ,test_size = 0.3 ,random_state = 0)


# In[19]:


#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(x_train , y_train)


# In[20]:


knn.score(x_train , y_train)


# In[21]:


predictions = knn.predict(x_test)


# In[22]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test , predictions)


# In[23]:


df_te.head()


# In[24]:


df_te.shape


# In[25]:


df_te = df_te.drop(['id'] , axis = 1)
df_te.shape


# In[26]:


test_pred = knn.predict(df_te)


# In[27]:


df_te['predicted_price'] = test_pred


# In[29]:


df_te.head()


# In[ ]:




