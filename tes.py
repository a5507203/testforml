
import numpy as np

mu, sigma = 0, 1 # mean and standard deviation
filename = 'data.txt'




def sampleFromNormal(size, mu, sigma):

    while True:
        x = np.random.normal(mu, sigma, size) 
        c1 = abs(sigma - np.std(x, ddof=1)) < 0.005
        c2 = (abs(mu - np.mean(x)) < 0.005)
        if (c1 and c2):
            return x
    


def addLabel(x1,x2, label):
    print(x1.shape[0])
    labels = np.full((x1.shape[0]), label,dtype=np.integer)
    return np.array((x1,x2,labels)).T

def saveToTxt(data, filename):
    np.savetxt(filename, data, fmt="%f")


x1 = sampleFromNormal(size = 1000, mu = 10, sigma = 10)
x2 = sampleFromNormal(size = 1000, mu = 0, sigma = 20)

data = addLabel(x1,x2,-1)
# print (data)

np.savetxt(filename, data, fmt="%f")
