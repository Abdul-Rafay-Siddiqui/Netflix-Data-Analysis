#!/usr/bin/env python
# coding: utf-8

# # Programming for Artificial Intelligence Project
# # BAI-3A
# ## 20K-1710, 20K-1712, 20K-1736
# 
# ### =======================================================
# 
# # Cleaning of Data:
# 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
def import_netflix():
    return pd.read_csv(r'C:\Users\pc\OneDrive\Desktop\netflix__titles.csv')

netflix = import_netflix()

print(netflix.shape)


# In[3]:


netflix.head()


# In[4]:


netflix.tail()


# In[5]:


print('Rows, Columns')
print(netflix.shape)

not_null_df = pd.DataFrame(netflix.notnull().sum(), columns=['Count of content'])
not_null_df.index.name = 'Column Name'
not_null_df[ not_null_df['Count of content']>0].sort_values('Count of content', ascending=False)


# In[ ]:





# In[6]:


null_df = pd.DataFrame(netflix.isnull().sum(), columns=['Count of Nulls'])
null_df.index.name = 'Column Name'
null_df[ null_df['Count of Nulls']>0].sort_values('Count of Nulls', ascending=False)


# In[7]:


netflix[['show_id', 'title']]


# In[8]:


netflix.dtypes


# In[9]:


netflix['date_added'] = pd.to_datetime(netflix.date_added)
netflix.dtypes


# In[10]:


netflix.fillna({'country': 'Unavailable', 'director': 'Unavailable'}, inplace=True)


# In[11]:


null_df = pd.DataFrame(netflix.isnull().sum(), columns=['Count of Nulls'])
null_df.index.name = 'Column Name'
null_df[ null_df['Count of Nulls']>0].sort_values('Count of Nulls', ascending=False)


# In[12]:


netflix[netflix.cast.isnull()]


# In[13]:


netflix[netflix.director == 'Toshiya Shinohara']


# In[31]:


netflix.loc[netflix['director'] == 'Toshiya Shinohara', 'release_year'] = netflix['date_added']
netflix[netflix.director == 'Toshiya Shinohara'].head()


# In[15]:


netflix.info()


# In[16]:


netflix[netflix.title == 'Squid Game']


# In[17]:


netflix.director.value_counts()


# In[18]:


content_per_director = pd.DataFrame(netflix.loc[(netflix['director']!='Unavailable')].director.value_counts())
content_per_director.describe().round(1)


# In[19]:


netflix['year_available'] = netflix.date_added.dt.year
items_per_year = netflix.year_available.value_counts().sort_index() 


# In[20]:


netflix.tail()


# # VISUALIZATION:

# In[21]:


typee=netflix['type'].value_counts()
print(typee)
plt.figure(figsize=[8,6])
sb.barplot(x=typee.values , y=typee.index)
plt.title("Number of movies and TV shows")
plt.xlabel("Number")
plt.ylabel("Type")
plt.show()


# In[22]:


genre=netflix['listed_in'].value_counts()
plt.figure(figsize=[15,40])
plt.title("Genre Identification",fontdict={'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel("Number",fontdict={'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.ylabel("Genre",fontdict={'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
sb.barplot(x=genre.values , y=genre.index)
plt.show()


# In[23]:


ratingss=netflix['rating'].value_counts()
plt.figure(figsize=[15,10])
plt.title("Ratings",fontdict={'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel("Number",fontdict={'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.ylabel("Type",fontdict={'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
sb.barplot(x=ratingss.values , y=ratingss.index)
plt.show()


# In[24]:


plt.figure(figsize=[15,10])
sb.countplot(x=netflix['type'] , hue=netflix['rating'])
plt.title("Comparsion of rating between movie and TV show")
plt.xlabel("Type")
plt.ylabel("Number")
plt.show()


# In[25]:


plt.figure(figsize=[20,10])
sb.countplot(x=netflix['type'] , hue=netflix['duration'])
plt.title("Duration with type",fontdict={'fontname': 'Monospace', 'fontsize': 120, 'fontweight': 'bold'})
plt.xlabel("Type")
plt.ylabel("Number")
plt.show()


# # MACHINE LEARNING:

# In[52]:


netflixx =netflix.copy()
netflixx = netflixx.apply(LabelEncoder().fit_transform)
netflixx.dtypes
netflixx.head(10)


# In[7]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[9]:


x = netflix.drop(['type'], axis = 1)
y = netflix['type']
ss = StandardScaler().fit(netflixx.drop('type', axis = 1))
X = ss.transform(netflixx.drop('type', axis = 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)
lr = LogisticRegression()
model = lr.fit(X_train, y_train)
prediction = model.predict(X_test)
print("Acc on train: ", lr.score(X_train, y_train)*100)
print("Acc on test: ", lr.score(X_test, y_test)*100)


# In[ ]:


rfc = RandomForestClassifier()
model1 = rfc.fit(X_train, y_train)
prediction1 = model1.predict(X_test)
print("Acc on train: ", rfc.score(X_train, y_train)*100)
print("Acc on test: ", rfc.score(X_test, y_test)*100)


# In[64]:


print(ss)


# In[ ]:




