import numpy as np
import numpy
import pandas

data = pandas.read_csv("bankdata.txt",header=None)

X = numpy.array(data.iloc[:,:4])
Y = numpy.array(data.iloc[:,4])



def alladin_ka_gini(groups,y):
              total = len(y)
              gini = 0
              for group in groups:
                     if len(group)==0: continue
                     group_size = len(group)
                     sum = 0
                     for c in numpy.unique(y):
                              class_count = 0
                              for child in group:
                                        if int(y[child])==int(c):
                                                    class_count += 1
                              ratio = class_count/group_size
                              sum = sum + (ratio*ratio)
                     gini += (1 - sum)*(group_size/total)
              return gini



def get_mid(X,Y):
       data_dick = {}
       gini = float("inf")
       for column in range(X.shape[1]):
                            data = X[:,column]
                            for maim in numpy.unique(data):
                                        divided = []
                                        l = numpy.reshape(numpy.argwhere(data<maim),(-1,))
                                        r = numpy.reshape(numpy.argwhere(data>=maim),(-1,))
                                        divided.append(l)
                                        divided.append(r)
                                        output = alladin_ka_gini(divided,Y)
                                        if output<gini:
                                            gini = output
                                            data_dick['value'] = maim
                                            data_dick['divided'] = divided
       return data_dick


def leaves(arr):
                classes = numpy.unique(arr)
                import collections
                collection = collections.Counter(arr)
                tree = {}
                for c in classes:
                         tree[c] = collection[c]/len(arr)
                return tree




def move(root_split,X,Y,md,d):
            groups = root_split.pop('divided')
            l = groups[0]
            r = groups[1]
            if len(l)==0 or len(r)==0:
                     passing_array = numpy.append(l,r)
                     passing_y = numpy.append(Y[l],Y[r])
                     root_split['l'] = root_split['r'] = leaves(passing_y)
            elif d>=md:
                     root_split['l'] = leaves(Y[l])
                     root_split['r'] = leaves(Y[r])
            else:
                     root_split['l'] = get_mid(X[l],Y[l])
                     root_split['r'] = get_mid(X[r],Y[r])
                     move(root_split['l'],X[l],Y[l],md,d+1)
                     move(root_split['r'],X[r],Y[r],md,d+1)




def run(X,Y):
    split_data = get_mid(X,Y)
    move(split_data,X,Y,3,1)
    print(split_data)



run(X,Y)
