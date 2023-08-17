#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('CC GENERAL.csv')


# In[3]:


data.head()


# In[4]:


data.describe().T


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


info = pd.DataFrame(data.describe())
info=info.iloc[0]
info.values.shape


# In[8]:


data.isnull().sum()


# In[9]:


data['MINIMUM_PAYMENTS'].isnull().sum()/data['MINIMUM_PAYMENTS'].shape[0] *100
data['MINIMUM_PAYMENTS'].dtype


# In[10]:


for col in data.columns:
    print(f'The percentage of null valus in each data is {data[col].isnull().sum()/data[col].shape[0]*100} % ')


# In[11]:


sns.pairplot(data)


# In[12]:


data.corr()


# In[13]:


sns.heatmap(data.corr())


# In[14]:


for col in data.columns:
    if col != 'CUST_ID':
        data[col].fillna(data[col].mean(),inplace=True)


# In[15]:


data


# In[16]:


data.drop('CUST_ID',inplace = True,axis = 1)


# In[17]:


data


# In[18]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# In[19]:


scaler = StandardScaler()
data_scale = scaler.fit_transform(data)


# In[20]:


data_normal = normalize(data_scale)


# In[21]:


data_normal = pd.DataFrame(data_normal)


# In[23]:


data_normal


# In[24]:


sse = {}
for k in range(1,10):
    kmeans = KMeans(n_clusters=k,max_iter=1000)
    K = kmeans.fit(data_normal)
    sse[k] = K.inertia_
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(list(sse.keys()),list(sse.values()))
ax.set_xlabel('K value')
ax.set_ylabel('SSE')
sse
                                                


# ## Silhouette Coefficient Method

# In[26]:


silhouette_scores = []
for k in range(2,11):
    silhouette_scores.append(silhouette_score(data_normal,KMeans(n_clusters=k).fit_predict(data_normal)))


# In[27]:


k = list(range(2,11))


# In[28]:


plt.bar(k,silhouette_scores)
plt.xlabel('total clusters')
plt.ylabel('Sihouette Score')


# ## Clearly the optimum number of clusters is 6

# In[30]:


kmeans = KMeans(n_clusters=6)
kmeans.fit(data_normal)


# In[31]:


pd.DataFrame(kmeans.fit_predict(data_normal)).value_counts()


# In[39]:


pca= PCA(n_components=2)
reduced_data = pca.fit_transform(data_normal)
X = pd.DataFrame(reduced_data)
X.columns= ['P1','P2']


# In[41]:


sns.jointplot(data=data,x=X['P1'],y= X['P2'],hue=kmeans.fit_predict(X))
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0],centroids[:,1],marker='o',color='w')
plt.title('K-means clustering on the credit card fraud dataset (PCA-reduced data)\n'
          'Centroids are marked with white circle')
plt.show()


# In[ ]:


# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .01     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = X['P1'].min() - 1, X['P1'].max() + 1
y_min, y_max = X['P2'].min() - 1, X['P2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Obtain labels for each point in mesh. Use last trained model.

# https://www.quora.com/Can-anybody-elaborate-the-use-of-c_-in-numpy
# https://www.geeksforgeeks.org/differences-flatten-ravel-numpy/
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = kmeans.predict(np.array(list(zip(xx.ravel(), yy.ravel()))))

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
# https://stackoverflow.com/questions/16661790/difference-between-plt-close-and-plt-clf
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),cmap=plt.cm.winter,
           aspect='auto', origin='lower')
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='o', s=10, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the credit card fraud dataset (PCA-reduced data)\n'
          'Centroids are marked with white circle')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()


# In[ ]:




