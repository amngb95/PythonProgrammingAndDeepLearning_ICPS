from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('CC.csv')



##Null values
nulls = pd.DataFrame(dataset.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

modifiedDataset = dataset.fillna(dataset.mean())

nulls1 = pd.DataFrame(modifiedDataset.isnull().sum().sort_values(ascending=False)[:25])
nulls1.columns = ['Null Count']
nulls1.index.name = 'Feature'
print(nulls1)

x = modifiedDataset.iloc[:,1:-1]
y = modifiedDataset.iloc[:-1]

wcss=[]
for i in range(1,7):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

print(wcss)
plt.plot(range(1,7),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)

'''X_scaled.sample(5)'''

from sklearn.cluster import KMeans

nclusters = 3 # this is the k in kmeans
seed = 0

km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(x)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(x)

from sklearn import metrics
score = metrics.silhouette_score(x, y_cluster_kmeans)
print('Silhouette Score is :',score)


from sklearn.decomposition import PCA
pca= PCA(2)
X_pca= pca.fit_transform(X_scaled)
print(X_pca)

km.fit(X_pca)
y_cluster_kmeans = km.predict(X_pca)
from sklearn import metrics
score = metrics.silhouette_score(X_pca, y_cluster_kmeans)
print('Silhouette Score After PCA :',score)