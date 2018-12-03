import numpy
from scipy import stats




A1= numpy.random.normal(1,1,50)
A2= numpy.random.normal(10,1,50)
A3= numpy.random.normal(1000,1,50)

A = numpy.array((A1,A2,A3))

B1= numpy.random.normal(1,1,50)
B2= numpy.random.normal(10,1,50)
B3= numpy.random.normal(1000,1,50)

B = numpy.array((B1,B2,B3))

#print(A)

C = stats.ttest_ind(A,B,1)
print(C)

print( stats.ttest_ind(A1,B1))
print( stats.ttest_ind(A2,B2))
print( stats.ttest_ind(A3,B3))
