import numpy
import pandas
class predefined:
    def relu(self,z):
         return (z>0)*z


    def sigmoid(self,z):
         sig = 1/(1+(numpy.exp(-z)))
         return sig


    def tanh(self,z):
         return numpy.tanh(z)


    def sigmoid_derivative(self,z):
           return self.sigmoid(z)*(1-self.sigmoid(z))


    def relu_derivative(self,z):
           return (z>0)*1


    def tanh_derivative(self,z):
          return 1-(numpy.pow(numpy.tanh(z),2))


    def squared_lose(self,y,hx):
          return (y-hx)**2


    def binary_cross_entropy_loss(self,y,hx):
          return -((y*numpy.log(hx))+((1-y)*numpy.log(1-hx)))


    def loss_derivative(self,y,hx):
          one = numpy.divide(y,hx)
          two = numpy.divide((1-y),(1-hx))
          return -(one-two)



class the_art_of():
    def __init__(self):
        self.functions = predefined()

    def inside(self,X,w1,w2,b1,b2):
        h1 = X.dot(w1)+b1
        o1 = self.functions.relu(h1)
        h2 = o1.dot(w2)+b2
        o2 = self.functions.sigmoid(h2)
        return {'o2':o2,'h2':h2,'o1':o1,'h1':h1}


    def outside(self,X,w1,w2,b1,b2,y,values,m,learningrate):
        clip = 0.01
        do2 = self.functions.loss_derivative(y,values['o2'])
        dh2 = do2*self.functions.sigmoid_derivative(values['h2'])
        dw2 = numpy.dot(values['o1'].T,dh2)/m
        db2 = numpy.sum(dh2,axis=0,keepdims=True)/m
        do1 = dh2.dot(w2.T)
        dh1 = do1*self.functions.relu_derivative(values['h1'])
        dw1 = numpy.dot(numpy.reshape(X,(1,2)).T,dh1)/m
        db1 = numpy.sum(dh1,axis=0,keepdims=True)/m
        w1 = w1 - numpy.clip(learningrate*dw1,-clip,clip)
        w2 = w2 - numpy.clip(learningrate*dw2,-clip,clip)
        b1 = b1 - (learningrate*db1)
        b2 = b2 - (learningrate*db2)
        return {'w1':w1,'w2':w2,'b1':b1,'b2':b2}


    def start_humping(self):
        data = pandas.read_csv("test.csv")
        X = data.iloc[:,0:2]
        Y = data.iloc[:,2:3]
        X = numpy.array(X)
        Y = numpy.array(Y)
        global x
        x = X[3]
        w1 = numpy.random.randn(2,3)
        w2 = numpy.random.randn(3,1)
        b1 = numpy.zeros(shape=(1,3))
        b2 = numpy.zeros(shape=(1,1))
        epoches = int(input("Enter number of epoches : "))
        for i in range(epoches):
                  for k in range(len(X)):
                           output = self.inside(X[k],w1,w2,b1,b2)
                           repaired = self.outside(X[k],w1,w2,b1,b2,Y[k],output,len(X[k]),1)
                           w1 = repaired['w1']
                           w2 = repaired['w2']
                           b1 = repaired['b1']
                           b2 = repaired['b2']
        return {'w1':w1,'w2':w2,'b1':b1,'b2':b2}


    def predict(self,weights,X):
              A1 = X.dot(weights['w1'])+weights['b1']
              Z1 = self.functions.relu(A1)
              A2 = Z1.dot(weights['w2'])+weights['b2']
              predicted = self.functions.sigmoid(A2)
              return predicted



ob = the_art_of()
weights = ob.start_humping()
print(x)
print(ob.predict(weights,x))
