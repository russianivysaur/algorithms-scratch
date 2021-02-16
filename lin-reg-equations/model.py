import math
import time
import pandas as pd
import threading
import matplotlib.pyplot as plt
import numpy
from sklearn.linear_model import LinearRegression
df = pd.read_csv("house.csv")
def data():
        global x
        global y
        y = df['Price']
        x = df['SqFt']
        

def repair(theta0,theta1,l):
       i = 0
       sum = 0
       for i in range(x.size):
               sum = sum + ((theta0 + (theta1*x[i])) - y[i])
       sum1 = sum/x.size;
       sum = 0
       i = 0
       for i in range(x.size):
               sum = sum + (((theta0+(theta1*x[i]))-y[i]) * x[i])
       sum = sum/x.size;  
       theta0 = theta0 - (l*sum1)
       theta1 = theta1 - (l*sum)
       return theta0,theta1


def run():
     theta0 = 0
     theta1 =0
     l =0.3
     i = 0
     for i in range(1000):
            theta0,theta1 = repair(theta0,theta1,l)
     model = LinearRegression()
     model.fit(numpy.array(x).reshape(-1,1),numpy.array(y).reshape(-1,1))
     print(theta0+(theta1*1000))
     print(model.predict([[1000]]))
     i  =0 
     preds = []
     for i in range(x.size):
             preds.append(model.predict([[x[i]]])[0])
     plt.scatter(x,y)
     plt.plot(x,theta0+(theta1*x))
     plt.plot(x,preds)
     plt.show()


data()
run()
