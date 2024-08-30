import csv
import numpy as np
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def polar(data):
    data_new=np.zeros(data.shape,dtype=float)
    for i in range(data.shape[0]):
        data_new[i]=[math.sqrt(data[i][0]**2+data[i][1]**2),math.atan(data[0][1]**2/data[0][0]**2)]
    return data_new

csv_file_path = 'C:/Users/Lenovo/Downloads/cluster.csv'

data = []
with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader, None)
    for row in csv_reader:
        data.append([float(row[0]), float(row[1])])

data=np.array(data)
data_polar=polar(data)
num_clusters = 2

kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(data_polar)
labels = kmeans.labels_

for i in range(num_clusters):
    cluster_points = data[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Raw Data and Clusters')
plt.legend()
plt.show()