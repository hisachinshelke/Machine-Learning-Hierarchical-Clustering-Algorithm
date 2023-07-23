#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Author: Sachin Baburao Shelke


# In[2]:


# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy.stats import zscore


# In[3]:


dataF = pd.read_csv("C:/Users/SK/Desktop/Class/Cust_Spend_Data.csv")


# In[4]:


dataF


# In[5]:


dataF_Attr = dataF.iloc[:,2:]


# In[6]:


dataF_Attr


# In[7]:


dataF_Attr_scaled = dataF_Attr.apply(zscore)


# In[8]:


dataF_Attr_scaled


# In[13]:


sns.pairplot(dataF_Attr_scaled, height=2, aspect=2, diag_kind='kde')


# In[16]:


from sklearn.cluster import AgglomerativeClustering


# In[17]:


model = AgglomerativeClustering(n_clusters=3 , affinity="euclidean" , linkage="average")


# In[18]:


model


# In[19]:


model.fit(dataF_Attr_scaled)


# In[20]:


model.labels_


# In[23]:


dataF_Attr["cluster_label"] = model.labels_


# In[27]:


dataF_Attr.drop(["labels"], axis=1)


# In[28]:


dataF_Attr[dataF_Attr['cluster_label']==0]


# In[29]:


dataF_Attr[dataF_Attr['cluster_label']==1]


# In[30]:


dataF_Attr[dataF_Attr['cluster_label']==2]


# In[31]:


dataF['cluster_label'] = model.labels_


# In[32]:


dataF


# In[33]:


dataF[dataF['cluster_label']==2]


# In[34]:


dataF[dataF['cluster_label']==1]


# In[35]:


dataF[dataF['cluster_label']==0]


# In[36]:


df_Cluster = dataF_Attr.groupby(['cluster_label'])


# In[37]:


df_Cluster


# In[38]:


df_Cluster.mean()


# In[39]:


from scipy.cluster.hierarchy import cophenet, dendrogram, linkage


# In[40]:


from scipy.spatial.distance import pdist


# In[41]:


dend_value = linkage(dataF_Attr_scaled, metric='euclidean', method="average")


# In[42]:


dend_value


# In[43]:


c , coph_dists = cophenet(dend_value, pdist(dataF_Attr_scaled))


# In[44]:


c


# In[45]:


pdist(dataF_Attr_scaled)


# In[48]:


plt.figure(figsize=(12,6))
plt.title("Agglomerative Hierarchical CLustering Dendogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
dendrogram(dend_value, leaf_rotation=30, leaf_font_size=10)
plt.tight_layout()


# In[49]:


dend_value2 = linkage(dataF_Attr_scaled, metric='euclidean', method="complete")


# In[50]:


c , coph_dists = cophenet(dend_value2, pdist(dataF_Attr_scaled))


# In[51]:


c


# In[52]:


plt.figure(figsize=(12,6))
plt.title("Agglomerative Hierarchical CLustering Dendogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
dendrogram(dend_value2, leaf_rotation=30, leaf_font_size=10)
plt.tight_layout()


# In[ ]:




