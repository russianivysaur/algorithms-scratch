import cmath
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import scipy.stats
import math
import matplotlib.pyplot as plt
import numpy
import pandas
data = pandas.read_csv("cancer.csv")
Y = data.iloc[:,1:2]
X = data.iloc[:,3:32]
Y = Y.replace('M',1)
Y = Y.replace('B',0)
x_ori = X
Y = numpy.array(Y)
X = numpy.array(X)
groups = []
means = []
classes = []
for i in range(Y.size):
          c = Y[i][0]
          presentbit = 0
          for k in range(len(classes)):
                     if classes[k]==c:
                             presentbit = 1
                             break
          if presentbit==0:
                  classes.append(c)


for i in range(len(classes)):
               data = []
               for k in range(Y.size):
                        if Y[k][0]==classes[i]:
                                        data.append(X[k])
               groups.append(data)


means = []
for i in range(len(classes)):
              meanvec = numpy.empty(len(x_ori.columns))
              for ss in range(meanvec.size):
                           meanvec[ss] = 0
              for k in range(len(groups[i])):
                           h = 0
                           for h in range(groups[i][k].size):
                                            meanvec[h] = meanvec[h] + groups[i][k][h]
              meanvec = meanvec/len(groups[i])
              means.append(meanvec)

overall_mean = numpy.empty(len(x_ori.columns))
for i in range(X.ndim):
          sum = 0
          for k in range(Y.size):
                      sum = sum+X[k][i]
          overall_mean[i] = sum/Y.size


sum = 0
for i in range(len(means)):
            difference = numpy.array([means[i]-overall_mean])
            n = len(means)
            sum = sum + (difference.T*n).dot(difference)

sb = sum

sum = 0
for i in range(len(means)):
          mean = means[i]
          for n in groups[i]:
                       x = numpy.array(n)
                       diff = numpy.array([x-mean])
                       sum = sum + (diff.T).dot(diff)


sw = sum


swdotsb = numpy.linalg.pinv(sw).dot(sb)
evals,evecs = numpy.linalg.eig(swdotsb)
maxeval = 0
for i in evals:
        if i>maxeval:
               maxeval = i

maxindex = 0
index = 0
for i in evals:
        if i==maxeval:
                 maxindex = index
        index = index+1

w = []
for k in range(evecs[0].size):
              w.append(evecs[k][maxindex])

w = numpy.array(w)
def guassian_distrib(a):
      arr = numpy.empty(a.size)
      index = 0
      for x in a:
            first = 1/(a.std()*math.sqrt(2*math.pi))
            num = -(x-a.mean())**2
            den = (2*a.std()*a.std())
            mid  = num/den
            second = math.exp(mid)
            res = first*second
            arr[index] = res
            index = index+1
      return arr



output_groups = []
for i in range(len(groups)):
           groups_list = []
           for k in range(len(groups[i])):
                      groups_list.append(groups[i][k].dot(w))
           output_groups.append(groups_list)


def draw(output_grps):
        x_min = -5.0
        x_max = 5.0
        mean_0 = numpy.array(output_grps[0]).mean()
        std_0 = numpy.array(output_grps[0]).std()
        mean_1 = numpy.array(output_grps[1]).mean()
        std_1 = numpy.array(output_grps[1]).std()
        x = numpy.linspace(x_min, x_max, 100)
        y_0 = scipy.stats.norm.pdf(x,mean_0,std_0)
        y_1 = scipy.stats.norm.pdf(x,mean_1,std_1)
        plt.plot(x,y_0)
        plt.plot(x,y_1)
        plt.xlim(x_min,x_max)
        plt.show()


def cal_threshold(outgrps):
                outgrp0 = numpy.array(outgrps[0])
                outgrp1 = numpy.array(outgrps[1])
                std1 = outgrp0.std()
                std2 = outgrp1.std()
                mean1 = outgrp0.mean()
                mean2 = outgrp1.mean()
                if(std1==std2):
                          threshold = (mean1+mean2)/2
                else:
                     one = 4/((std1**2) * (std2**2))
                     two = (mean1-mean2)**2
                     three = (std1**2)-(std2**2)
                     four = math.log((std1**2)/(std2**2))
                     d = one*(two+(three*four))
                     a = (-1/(std1**2))+(1/(std2**2))
                     b = 2*((-mean2/(std2**2))+(mean1/(std1**2)))
                     root1 = (-b+math.sqrt(d))/(2*a)
                     root2 = (-b-math.sqrt(d))/(2*a)
                     print(root2)

draw(output_groups)
cal_threshold(output_groups)
outputs = []
for i in range(Y.size):
             outputs.append(complex(X[i].dot(w)).real)
outputs = numpy.array(outputs).reshape(-1,1)
print(outputs)

X_train,X_test,Y_train,Y_test = train_test_split(outputs,Y,test_size=0.5,shuffle=True)
X_train = numpy.array(X_train).reshape(-1,1)
X_test = numpy.array(X_test).reshape(-1,1)
model = LogisticRegression()
model.fit(X_train,Y_train)
print(model.score(X_test,Y_test))
