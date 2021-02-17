import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
data = pd.read_csv("cancer.csv")



def sigmoid(x):
         return 1/(1+math.exp(-x))

def funtion(x,theta0,theta1):
         return sigmoid(theta0+(theta1*x))


def repair(X,Y,theta0,theta1,l):
              m = Y.size
              sum = 0
              sum1 = 0
              sum2 = 0
              for i in range(m):
                      sum = sum+(funtion(X[i],theta0,theta1)-Y[i])
              sum1 = sum1/m
              sum = 0
              for i in range(m):
                        sum = sum+((funtion(X[i],theta0,theta1)-Y[i])*X[i])
              sum2 = sum/m
              theta0 = theta0 - (l*sum1)
              theta1 = theta1 - (l*sum2)
              return theta0,theta1 


Y = data['diagnosis']
X = data.iloc[:,2:32]
X = X.fillna('0')
Y = Y.replace(to_replace='B',value=1)
Y = Y.replace(to_replace='M',value=0)
X = numpy.array(X)
Y = numpy.array(Y)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0,shuffle=True)
lda = LinearDiscriminantAnalysis()
X_train = lda.fit_transform(X_train,Y_train)
X_test = lda.fit_transform(X_test,Y_test)
model = LogisticRegression()
model.fit(X_train,Y_train)
X = numpy.array(X_train)
Y = numpy.array(Y_train)
theta0 = numpy.random.rand()
theta1 = numpy.random.rand()
l = 0.3
for i in range(2000):
            theta0,theta1 = repair(X,Y,theta0,theta1,l)
i = 0
X_zero_pred = []
X_one_pred = []
for i in range(Y_test.size):
            model_pred = model.predict([X_test[i]])
            if(model_pred!=Y_test[i]):
                      print("model miss")
            algo_pred = (theta0+(theta1*X_test[i]))
            if algo_pred >= 0.5:
                    algo_pred = 1
                    X_one_pred.append(X_test[i])
            else:
                    algo_pred = 0
                    X_zero_pred.append(X_test[i])
            if(model_pred[0]!=Y_test[i]):
                            print("missed")

X_one = []
X_zero = []
for j in range(Y_test.size):
        if(Y_test[j]==0):
              X_zero.append(X_test[j])
        elif(Y_test[j]==1):
              X_one.append(X_test[j])

plot = plt.figure().gca(projection='3d')
plot.scatter(numpy.array(X_zero),2,0,color='greenyellow')
plot.scatter(numpy.array(X_one),2,0,color='red')
plot.scatter(numpy.array(X_zero_pred),3,0,color='darkolivegreen')
plot.scatter(numpy.array(X_one_pred),3,0,color='maroon')
plt.show()
