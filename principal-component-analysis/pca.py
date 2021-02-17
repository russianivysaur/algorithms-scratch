import numpy
import pandas
data = pandas.read_csv("cancer.csv")
Y = data.iloc[:,1:2]
X = data.iloc[:,3:32]
Y = Y.replace('M',1)
Y = Y.replace('B',0)
means = numpy.mean(X,axis=0)
controlled_X = X - means
covariance = numpy.cov(controlled_X)
eigval , eigvec = numpy.linalg.eig(covariance)
max = 0;
sum = 0;
for i in range(eigval.size):
              sum = sum+eigval[i];
              if max<eigval[i]:
                      max = eigval[i]

print(sum)
print(max)
print((max/sum)*100)
