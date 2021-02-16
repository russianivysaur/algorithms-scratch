import numpy as np
import numpy
import collections
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
X = np.array( [ ['A','N','E'], ['A','M','E'], ['B','M','G'], ['B','L','F'],
                ['A','K','G'], ['B','L','E'], ['A','M','G'], ['B','N','F'], ['B','L','G']])
Y = np.array( [0,0,0,0,1,1,1,1,1] )
for i in range(len(X[0])):
        X[:,i] = LabelEncoder().fit_transform(X[:,i])
X = X.astype('uint8')
print(X)
cats = np.unique(Y)
probs = []
for i in range(len(cats)):
          sum = 0
          for k in range(len(Y)):
                   if(Y[k]==cats[i]):
                        sum = sum+1
          probs.append(sum/len(Y))
print(X[4])

def get_probabilities():
          dictionaries = []
          for i in range(len(X[0])):
                       dictionaries.append({})
          for i in range(len(cats)):
                      flagged = []
                      category = cats[i]
                      for k in range(len(Y)):
                             if(Y[k]==cats[i]):
                                     flagged.append(X[k])
                      flagged = np.array(flagged)
                      for s in range(len(flagged[0])):
                                  column = flagged[:,s]
                                  total = len(column)
                                  uniques = np.unique(column)
                                  counter = collections.Counter(column)
                                  count = len(column)
                                  for h in range(len(uniques)):
                                                parent = str(category)
                                                child = str(uniques[h])
                                                prob_house = parent+"to"+child
                                                uniquescount = counter[uniques[h]]
                                                prob = uniquescount/total
                                                dictionaries[s][prob_house] = prob
          return dictionaries



p = get_probabilities()
print(p)
x = X[4]
classes = np.unique(Y)
finalprobs = []
for i in range(len(classes)):
              parent = str(classes[i])
              pr = probs[i]
              for s in range(len(x)):
                        child = str(x[s])
                        combined = parent+"to"+child
                        try:
                          pr = pr*p[s][combined]
                        except KeyError:
                          pr = pr*1
              finalprobs.append(pr)

finalprobs = np.array(finalprobs)
finalprobs = finalprobs/np.sum(finalprobs)
print(finalprobs)
print("Predicted class : ",np.argmax(finalprobs))
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
model = MultinomialNB()
model.fit(X,Y)
print("Predicted class : ",model.predict(x.reshape(1,-1)))
print(model.score(X,Y))
model = BernoulliNB()
model.fit(X,Y)
print("Predicted class : ",model.predict(x.reshape(1,-1)))
print(model.score(X,Y))
