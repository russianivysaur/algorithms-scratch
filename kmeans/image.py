import cv2
import numpy
img = cv2.imread("image.jpg")
img = cv2.resize(img,(400,400))
cv2.imwrite("image2.jpg",img)
img = img/255
cv2.imwrite("image3.jpg",img)
data = numpy.reshape(img,(img.shape[0]*img.shape[1],img.shape[2]))

import pandas
import numpy
import math
X = data
X = numpy.array(X)
clusters = 5
centers = []
for i in range(clusters):
        rand = numpy.random.randint(400*400)
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



for k in range(10):
            print("step ",k)
            cluster_dumps = make_clusters()
            for i in range(len(centers)):
                         array = numpy.array(cluster_dumps[i])
                         mean_vec = numpy.zeros(shape=(1,3))
                         for s in range(len(array)):
                                      mean_vec = mean_vec + array[s]
                         mean_vec = mean_vec/len(array)
                         centers[i] = mean_vec

def get_center(centers,point):
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
               return centers[index]


for i in range(len(X)):
          test = X[i]
          X[i] = get_center(centers,test)

X = X*255
X = X.astype("uint8")
im = X.reshape(400,400,3)
cv2.imwrite("5_cluster.jpg",im)
