import time
from sklearn.cluster import KMeans
import pandas
import numpy
import matplotlib.pyplot as plt
import math

fig = plt.figure()
g = fig.gca(projection='3d')
data = pandas.read_csv("iris.csv")
colors = ['orange','green','blue']
a = 0
for i in range(3):
        X1 = data.iloc[a:a+50,0:1]
        X2 = data.iloc[a:a+50,1:2]
        g.scatter(X1,X2,color=colors[i],s=10)
        a = a+50



X = data.iloc[:,0:2]
X = numpy.array(X)
clusters = 3
centers = []
for i in range(clusters):
        rand = numpy.random.randint(149)
        centers.append(X[rand])

def assign_center(centers,point):
               index = 0
               mindis = 9999
               for i in range(len(centers)):
                           sum = 0
                           for k in range(point.size):
                                         diff = (centers[i][k] - point[k])**2
                                         sum = sum+diff
                           dis = math.sqrt(sum)
                           if dis<mindis:
                                   mindis = dis
                                   index = i
               return index



centers = numpy.array(centers)

def make_clusters():
       cluster_dumps = []
       for k in range(len(centers)):
             list = []
             for i in range(len(X)):
                       if k==assign_center(centers,X[i]):
                                   list.append(X[i])
             cluster_dumps.append(list)
       return cluster_dumps


points = []
for i in range(len(centers)):
                        point = g.scatter(centers[i][0],centers[i][1],s=60,color='red',marker='^')
                        points.append(point)
plt.ion()
plt.show()
input = input("Enter 1 to start 0 to exit :")
if int(input)==0:
       exit(0)
for k in range(1000):
            time.sleep(0.8)
            cluster_dumps = make_clusters()
            for i in range(len(centers)):
                         array = numpy.array(cluster_dumps[i])
                         mean_vec = numpy.zeros(shape=(1,2))
                         for s in range(len(array)):
                                      mean_vec = mean_vec + array[s]
                         mean_vec = mean_vec/len(array)
                         centers[i] = mean_vec
            for i in range(len(centers)):
                        point = g.scatter(centers[i][0],centers[i][1],color='red',s=60,marker='^')
                        points[i].remove();
                        points[i] =point; 
                        fig.canvas.draw()
