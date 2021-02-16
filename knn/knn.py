from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy
import math

def distance(sample,x):
        dis = None
        sum = 0
        for (i,j) in zip(sample,x):
                sum  = sum + math.pow((i-j),2)
        dis = math.sqrt(sum)
        return dis


def find(X,Y,x,K,class_count):
        dis = []
        clas = []
        dtype = [('class','S15'),('dis',float)]
        for i,pred in zip(X,Y):
             dis.append((pred[0],distance(x,i)))
        dis = numpy.array(dis,dtype=dtype)
        dis = numpy.sort(dis,order='dis')
        k_list = []
        class_names = []
        class_names.append(Y[0][0])
        for i in range(Y.size):
                   present = 0
                   for k in range(len(class_names)):
                                    if class_names[k]==Y[i][0]:
                                               present = 1
                   if present==0:
                           class_names.append(Y[i][0])
        class_points = numpy.zeros(len(class_names))
        for l in range(K):
                k_list.append(dis[l])
                for j in range(class_points.size):
                               if dis[l][0].decode('utf-8')==class_names[j]:
                                                     class_points[j] = class_points[j]+1
                                                     break
        for h in range(len(class_points)):
                   class_points[h] = class_points[h] / K;
        return class_names,class_points



data = pd.read_csv("iris.csv")
X = data.iloc[:,0:4]
Y = data.iloc[:,4:5]
ax = plt.axes(projection='3d')
colors = ['green','yellow','blue']
i = 0
a = 0
while i<=100: 
        x_coords = numpy.array(data.iloc[i:i+50,2:3])
        y_coords = numpy.array(data.iloc[i:i+50,0:1])
        z_coords = numpy.array(data.iloc[i:i+50,3:4])
        ax.scatter3D(x_coords,y_coords,z_coords,colors[a])
        i = i+50
        a = a+1
X = numpy.array(X)
Y = numpy.array(Y).reshape(Y.size,1)
ax.scatter3D(x_coords[0],y_coords[0],z_coords[0],color='red')
plt.show()
model = KNeighborsClassifier(metric='euclidean',n_neighbors=5)
model.fit(X,Y)
i = 0
missed = 0
for i in range(Y.size):
               pred = str(model.predict([X[i]]))
               manualpred = find(X,Y,X[i],5,3)
               model_pred = numpy.array(model.predict_proba([X[i]])[0])
               algo_pred = numpy.array(find(X,Y,X[i],5,3)[1])
               if not numpy.array_equal(model_pred,algo_pred):
                         print(model_pred)
                         print(algo_pred)
                         missed = missed+1
print("Missed predictions:  ",missed)
print("Sklearn model accuracy : ",model.score(X,Y))
