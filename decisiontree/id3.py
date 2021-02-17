import numpy
import pandas
import math
import collections
data = pandas.read_csv("tennis.csv")
Y = data['PlayTennis']
X = data.iloc[:,2:6]
global_columns = X.columns
global_X = X
tree_depth = 2
loop_control = 0
global_Y = Y
X = numpy.array(X)
Y = numpy.array(Y)
def entropy(p):
       total = len(p)
       uniques = numpy.unique(p)
       collection = collections.Counter(p)
       e = 0
       for i in range(len(uniques)):
               count = collection[uniques[i]]
               weight = count/total
               other_classes = total - count
               e = e + (weight*(math.log(weight,2)))
       return -e


global_entropy = entropy(Y)
initial_sequence = numpy.arange(len(X[0]))
def get_split_entropies(data,Y):
                 uniques = numpy.unique(data)
                 sub_nodes = []
                 sub_node_indexes = []
                 for k in range(len(uniques)):
                          list = []
                          indexes = []
                          for g in range(len(data)):
                                     if(data[g]==uniques[k]):
                                                 list.append(Y[g])
                                                 indexes.append(g)
                          sub_nodes.append(list)
                          sub_node_indexes.append(indexes)
                 entropies = []
                 for s in range(len(uniques)):
                            entropies.append(entropy(numpy.array(sub_nodes[s])))
                 return numpy.array(entropies),numpy.array(sub_nodes),numpy.array(sub_node_indexes)



def get_minimum(X,Y,initial_sequence,global_entropy):
                  ig = numpy.arange(len(initial_sequence)).astype("float")
                  X = numpy.array(X)
                  node_indexes = []
                  sub_nodes = []
                  for i in range(len(initial_sequence)):
                                      sum = 0
                                      data = X[:,i]
                                      entropies,nodes,n = get_split_entropies(data,Y)
                                      node_indexes.append(n)
                                      sub_nodes.append(nodes)
                                      for j in range(len(nodes)):
                                                    sum = sum + ((len(nodes[j])/len(X))*entropies[j])
                                      gain = float(global_entropy - sum)
                                      ig[i] = gain
                  return ig,node_indexes,sub_nodes



def sub_atomic(data_indexes,feature,tree,columns,preceding,X,Y):
          global loop_control
          features = pandas.DataFrame()
          targets = []
          presence = numpy.zeros(4)
          for i in range(len(data_indexes)):
                         features[i] = X[data_indexes[i]]
                         targets.append(Y[data_indexes[i]])
          features = features.transpose()
          features.columns = columns
          child = features[feature][0]
          features = features.drop(feature,axis=1)
          e = entropy(targets)
          initial_sequence = numpy.arange(len(features.columns))
          if e!=0:
                preceding.append(child)
                run(features,targets,initial_sequence,e,tree,features.columns,preceding)
          elif e==0:
                copy = tree
                for i in range(len(preceding)):
                             tree = tree[preceding[i]]
                tree[child] = targets[0]
                return copy





def run(X,Y,initial_sequence,global_entropy,tree,columns,preceding):
         global loop_control
         s,indexes,subs = get_minimum(X,Y,initial_sequence,global_entropy)
         X = numpy.array(X)
         max = numpy.argmax(s)
         node = columns[max]
         all_values = global_X[node]
         values = numpy.array(all_values)
         branches = numpy.unique(values)
         copy = tree
         if preceding is None:
             preceding = []
         else:
             for i in range(len(preceding)):
                           tree = tree[preceding[i]]
         les = numpy.unique(numpy.array(global_Y))
         if tree not in les:
               tree[node] = {}
               for i in range(len(branches)):
                   tree[node][branches[i]] = {}
               preceding.append(node)
               for g in range(len(indexes[max])):
                          loop_control = loop_control+1
                          tree = sub_atomic(indexes[max][g],node,copy,columns,preceding,X,Y)
                          loop_control = loop_control-1
                          if(loop_control==0):
                                       ss = preceding[0]
                                       preceding=[]
                                       preceding.append(ss)



tree = {}
run(X,Y,initial_sequence,global_entropy,tree,global_columns,None)
print(tree)
