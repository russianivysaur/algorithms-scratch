from sklearn.preprocessing import StandardScaler
import numpy
import random
import pandas
import matplotlib.pyplot as plt
fig = plt.figure()
g = fig.gca(projection='3d')
data  = pandas.read_csv("iris.csv")
X = data.iloc[:,0:2]
Y = data.iloc[:,4:5]
X = numpy.array(X)
Y = numpy.array(Y)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
uniques = numpy.unique(Y)
x = list()
y = numpy.array([])
classes = [uniques[0],uniques[len(uniques)-1]]
for i in range(len(Y)):
       for s in range(len(classes)):
              if Y[i]==classes[s]:
                 l = list(X[i])
                 l.append(1)
                 x.append(numpy.array(l))
                 if s==0:
                    y = numpy.append(y,1)
                 else:
                    y = numpy.append(y,-1)
data = [[],[]]
for i in range(len(x)): data[0].append(x[i][0])
for i in range(len(x)): data[1].append(x[i][1])
a = 0
for i in range(2):
         g.scatter(numpy.array(data[0])[a:a+50],numpy.array(data[1])[a:a+50],2)
         a = a+49
weights = numpy.zeros(len(x[0]))
learning_rate = 0.000001
def adjust(X,Y,w,alpha):
         batch_derivative = numpy.zeros(len(X[0]))
         for x,y in zip(X,Y):
                 line_equation = 1 - (y*(x.dot(w)))
                 if(max(line_equation,0)==0):
                         loss_gradient = w
                 else:
                         loss_gradient = w - (alpha*y*x)
                 batch_derivative = batch_derivative+loss_gradient
         batch_derivative = batch_derivative/len(Y)
         return batch_derivative
x = numpy.array(x)
y = numpy.array(y)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.17,shuffle=True)
batch_size = 1
for i in range(24000):
         batch_shuffle = list(zip(xtrain,ytrain))
         random.shuffle(batch_shuffle)
         batch_shuffle = numpy.array(batch_shuffle,dtype='object')
         batch_shuffle = batch_shuffle[0:batch_size]
         batch_derivative = adjust(batch_shuffle[:,0],batch_shuffle[:,1],weights,1000)
         weights = weights - (learning_rate*batch_derivative)
final = []
from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(xtrain,ytrain)
acc = 0
for x,y in zip(xtest,ytest):
           res = x.dot(weights)
           if res>0 and y==1:
                acc += 1
           elif res<0 and y==-1:
                acc += 1
acc = acc/len(ytest)
print("Manuel model accuracy : ",acc)
print("w1 : ",weights[0])
print("w2 : ",weights[1])
print("intercept : ",weights[2])
print("\n\nSklearn model accuracy : ",model.score(xtest,ytest))
print("w1 : ",model.coef_[0][0])
print("w2 : ",model.coef_[0][1])
print("intercept : ",model.intercept_[0])
#extra stuff for visualizing
f = []
n = []
a = 1
for i in xtrain:
       f.append((i.dot(model.coef_[0])+model.intercept_)[0])
       n.append(a)
       a = a+1
g.plot(numpy.array(f),numpy.array(n),3)
f = []
n = []
a = 1
for i in xtrain:
       f.append(i.dot(weights))
       n.append(a)
       a = a+1
g.plot(f,n,4)
plt.show()
