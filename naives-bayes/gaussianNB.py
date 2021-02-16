import numpy
import numpy as np
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
data = pandas.read_csv("iris.csv")
X = data.iloc[:,0:4]
Y = data.iloc[:,4:5]
X = numpy.array(X)
encoder = LabelEncoder()
Y = numpy.array(encoder.fit_transform(Y))
X,xtest,Y,ytest = train_test_split(X,Y,test_size=0.3,shuffle=True)
cats = np.unique(Y)
cat_values = []
for i in range(len(cats)):
         cnt = 0
         for k in range(len(Y)):
                    if(Y[k]==cats[i]):
                          cnt = cnt+1
         arr = numpy.empty([cnt,len(X[0])])
         s = 0
         for k in range(len(Y)):
                    if(Y[k]==cats[i]):
                          arr[s] = X[k]
                          s=s+1
         mean = arr.mean(axis=0)
         std = arr.std(axis=0)
         cat_values.append([mean,std,cnt])


def proba(x,mean,std):
        a = 1/(std*numpy.sqrt(2*numpy.pi))
        b = numpy.exp(-0.5*(((x-mean)/std)**2))
        return a*b


def find_all_probas(x,Y,cats,cat_values):
              finalprobs = []
              for i in range(len(cat_values)):
                          p = cat_values[i][2]/len(Y)
                          for f in range(len(x)):
                                p = p*proba(x[f],cat_values[i][0][f],cat_values[i][1][f])
                          finalprobs.append(p)
              return finalprobs


def predict(x):
   probs = find_all_probas(x,Y,cats,cat_values)
   sum = 0
   for i in range(len(probs)):
              sum = sum+probs[i]
   probs = probs/sum
   return np.argmax(probs)

s = 0
for i in range(len(xtest)):
            if(predict(xtest[i])==ytest[i]):
                s = s+1

print("Test accuracy = ",s/len(xtest))
