import numpy
import math
import matplotlib.pyplot as plt
import pandas
data = pandas.read_csv("iris.csv")
Y = data.iloc[:,4:5]
X = data.iloc[:,0:2]
ini_X = X
X = list(numpy.expand_dims(X,axis=1))
print("read data....")
n=2
x = [[0.40,0.53],[0.22,0.38],[0.35,0.32],[0.26,0.19],[0.08,0.41],[0.45,0.30]]
x = list(numpy.expand_dims(x,axis=1))
cs = ['red','green','blue']
a = 0
for i in range(3):
         plt.scatter(data.iloc[a:a+50,0:1],data.iloc[a:a+50,1:2],color=cs[i])
         a = a+50
plt.show()
def euclidean(x1,x2):
          if(len(x1)!=len(x2)):
              print("malformed")
              exit(0)
          else:
              sum = 0
              for (i,j) in zip(x1,x2):
                    sum = sum + math.pow((i-j),2)
              sum = math.sqrt(sum)
              return sum

def distance(cluster1,cluster2):
              distances = []
              for i in cluster1:
                      for j in cluster2:
                               dis = euclidean(i,j)
                               distances.append(dis)
              return distances

def linkage(dises,code):
          if(code==0):
              #single
              min = float('inf')
              for i in dises:
                      if(min>i):
                           min = i
              return min

def make_proximity_matrix(input):
          all = []
          for i in input:
                   sub = []
                   for j in input:
                         dises = distance(i,j)
                         sub.append(linkage(dises,0))
                   all.append(sub)
          return numpy.mat(all)


def get_minimum(matrix):
            min = float('inf')
            i = int(math.sqrt(matrix.size))
            for row in range(i):
                    for column in range(i):
                               if(min>matrix[row,column]):
                                         if(row!=column):
                                               min = matrix[row,column]
                                               r = row
                                               c = column
            return (r,c)



def begin(input):
          clusters = input
          matrix = make_proximity_matrix(clusters)
          while(len(clusters)>n):
                     (i,j) = get_minimum(matrix)
                     new = []
                     for s in range(len(clusters)):
                                   if(s==i or s==j):
                                          continue
                                   else:
                                        new.append(clusters[s])
                     new.append(numpy.append(clusters[i],clusters[j],axis=0))
                     clusters = new
                     matrix = make_proximity_matrix(clusters)
          return clusters

import matplotlib.pyplot as plt
clusters = begin(X)
print(clusters)
for i in clusters:
        print("\n\n\n\n\n",i)
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='single')
print(model.fit_predict(ini_X))
