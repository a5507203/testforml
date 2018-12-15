import numpy as np
import csv
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from numbers import Number
from scipy import stats
import random
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from collections import defaultdict
import os
import errno
from textwrap import wrap

class Metamorphic_Test:

    def __init__(self, url= './data/iris.data', classifierType = 'knn'):
        
        self.df, self.indices, self.uniqueLabels = self.getData(url)
        self.classifierType = classifierType
        self.classifier = self.select_classifier(classifierType, self.df)
        self.result_dic = defaultdict(dict)
        # self.lOOCV_results = []
        # self.mr_results = []

    def select_classifier(self, classifierType, df):
        attributes, labels = self.convertDataFormat(df)
        if classifierType == 'svm':
            return svm.SVC(C=1.0, kernel='linear').fit(attributes,labels)
        elif classifierType == 'linearSVM':
            return svm.LinearSVC(random_state=0, tol=1e-5).fit(attributes,labels)
        elif classifierType == 'knn':
            return KNeighborsClassifier(n_neighbors=1).fit(attributes,labels)
        elif classifierType == 'gaussianNB':
            return GaussianNB(priors=None, var_smoothing=1e-09).fit(attributes,labels)
        elif classifierType == 'multinomialNB':
            return MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True).fit(attributes,labels)
        else:
            return None


    def getData(self, url ):
        df = pd.read_csv(url)
        #change the value of the attribute to numberical value if is not
        enc = LabelEncoder()

        # df = df.sample(frac=0.75)
        indices = []
        for i in range (len(df.columns)):

            indices.append(i)
            col = df.columns[i]

            if(isinstance(df[col].iloc[0], Number)):
                continue
            enc.fit(df[col])
            df[col] = enc.transform(df[col])

        uniqueLabels = df[df.columns[indices[-1]]].unique()


        return df, indices, uniqueLabels


    def convertDataFormat(self, df):

        attriutes = df.take(self.indices[0:-1], axis=1).values.tolist()
        labels = df.take([self.indices[-1]], axis=1).values.flatten()
        return attriutes, labels

    # attributeIndex represents which attribute is required
    def prepareFormatForTTest(self, df, attributeIndex = None ):
        if attributeIndex == None:
            return df.take(self.indices[0:-1], axis=1).T.values.tolist()
        else:
            return df.take([self.indices[attributeIndex]], axis=1).T.values.tolist()


    def getAllInstanceBylabel(self, label):
        return self.df[self.df['class'] == label]

    def getColumn(self, df, index):
        return df[df.columns[index]]


    def lOOCV(self, label2):
        predictions = []
        indices = self.df.index[self.df['class'] == label2].tolist()

        for index in indices:
            testDf = self.df.iloc[[index]].copy()
            trainingDf = self.df.drop(self.df.index[[index]])

           
            classifier = self.select_classifier(self.classifierType,trainingDf)
            testAttr, testLabel = self.convertDataFormat(testDf)
            prediction = self.predict(classifier, testAttr)[0]
            if prediction != label2:
                predictions.append(0)
            else:
                predictions.append(1)
        return predictions
            # print(b)

    def metamorphic_Test_continious_attr(self, label1, label2, attributeIndex, mu = 0, sigmas = [1,2,3,4,5,6], repeatForEverySigma = 20 ):

        df1 = self.getAllInstanceBylabel(label1).copy()
        df2 = self.getAllInstanceBylabel(label2).copy()
      
        # print(len(df1))
        # print(df2)
        repeatTime = len(sigmas)*repeatForEverySigma
        # print(df2)

        i = 0
        predictions = np.empty(shape = [len(df2),repeatTime], dtype=int)
        for sigma in sigmas:
            for j in range(repeatForEverySigma):
                df3 = df2.copy()
                df3[df3.columns[attributeIndex]] += np.random.normal(mu, sigma, len(df3)) 

        
                attri, label = self.convertDataFormat(df3)
                prediction = self.predict(self.classifier,attri)
                predictions[:,i] = prediction
                i+=1

        results = []
        for i in predictions:
            result = ((i == label2).sum())
            results.append(1-float(result)/float(repeatTime) )
        return(results)
      
      
    # def swapAttribute(self, label1, label2, attributeIndex):

    #     df1 = self.getAllInstanceBylabel(label1)
    #     df2 = self.getAllInstanceBylabel(label2)

    #     col1 = self.getColumn(df1,attributeIndex)
    #     col2 = self.getColumn(df2,attributeIndex)

    #     # temp = col1
    #     # col1 = col2
    #     # col2 =temp
    #     print (df1)
    #     print(df2)

    #     temp = col1
    #     col1 = col2
    #     col2 =temp
    #     print (df1)
    #     print(df2)
    #     self.t_test(df1,label1,df2,label2, attributeIndex)

    # def permutationDiscreteAttribute(self, label1, label2, attributeIndex):

    #     df1 = self.getAllInstanceBylabel(label1)
    #     df2 = self.getAllInstanceBylabel(label2)

    #     # print(df2)
    #     self.t_test(df1,label1,df2,label2, attributeIndex)
    #     # df2[df2.columns[attributeIndex]] = np.random.permutation(df2[df2.columns[attributeIndex]] )
    #     # self.t_test(df1,label1,df2,label2, attributeIndex)
    #     # print(df2)
    #     attri, label = self.convertDataFormat(df2)
    
    #     a= self.predict(attri)

    def tTests(self):
            
        dfs = []
        for label in self.uniqueLabels:
            # print(self.df[self.df['class'] == label])
            dfs.append(self.df[self.df['class'] == label])
   
        for i in range(len(self.uniqueLabels)):
            df1 = dfs[i]
            for j in range(i+1,len(dfs)):
                df2 = dfs[j]
                self.t_test(df1,self.uniqueLabels[i],df2,self.uniqueLabels[j])
        
        self.print_results()


    def print_results(self):

        print("number of attributes: ",len(self.indices)-1)
        print("number of labels: ",len(self.uniqueLabels))
        print("number of samples: ",len(self.df))

        print(' ')
        print('__________________________test_result_____________________________')

        print(' ')
        print('____________________t result between class labels i and j_______________________')
        for label_i in sorted(self.result_dic):
            for label_j in sorted(self.result_dic[label_i]):
                print(' ')
                print('___________________________')
                print('t_value between ', label_i, ' and ',label_j, ': ',self.result_dic[label_i][label_j].statistic)


        print(' ')
        print(' ')
        print(' ')
        print('__________________________p result between class labels i and j_____________________________')

        pvalue_results_list = []
    
        for label_i in sorted(self.result_dic):
            for label_j in sorted(self.result_dic[label_i]):
                print(' ')
                print('___________________________')
                print('p_value between ', label_i, ' and ',label_j, ': ',self.result_dic[label_i][label_j].pvalue)
                pvalue_results_list.append(self.result_dic[label_i][label_j].pvalue)


        results = np.array(pvalue_results_list)

        print('')
        print('')
        print('')
        print('__________________p value Summary_____________________')
        print(' ')  
        print("cross all labels, mean and std of the p value for all attribute pairs are:")
        print('mean: ', results.mean(axis=0))
        print(' std: ', results.std(axis=0))
        print(' ')  

        
    def t_test(self,df1,label_i,df2,label_j, attributeIndex = None):

        df1_attributes = self.prepareFormatForTTest(df1,attributeIndex)
        df2_attributes = self.prepareFormatForTTest(df2,attributeIndex)

        result = stats.ttest_ind(a = df1_attributes,b = df2_attributes, axis = 1 )

        self.result_dic[label_i][label_j] = result
        print(result)

    def predict(self,classifier, data):
        return classifier.predict(data)


def createDictionary(path):

    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":

    
    name = 'sensor_readings_24'
    name = 'Frogs_MFCCs'
    data_url = './data_and_results/'+name+'/'+name+'.data'

    classifierType = 'gaussianNB'
    classifierType = 'multinomialNB'
    classifierType = 'linearSVM'
    classifierType = 'svm'
    classifierType = 'knn'
    label1 = 1
    label2 = 2
    attributeIndex = 14
    sigmas = [0.1,0.2,0.3]
    p_value = 3.67748857e-064
    repeatForEverySigma = 100

    np.set_printoptions(linewidth = 500)

    #write to file
    #sys.stdout = open(name+'_t_test_result.txt','wt')

    print('dataset_name: ',name)
    print ('data_url: ',data_url)

    # # select classifier

    metamorphic_Test = Metamorphic_Test(url = data_url, classifierType = classifierType)
    #metamorphic_Test.tTests()

    mr_result = metamorphic_Test.metamorphic_Test_continious_attr(label1 = label1, label2 = label2, attributeIndex = attributeIndex, mu = 0, sigmas=sigmas,repeatForEverySigma=repeatForEverySigma)
    loocv_result = metamorphic_Test.lOOCV(label2=label2)


    #### write test result to file

    createDictionary('./data_and_results/'+name+'/'+classifierType)

    fileName = './data_and_results/'+name+'/'+classifierType+'/'+name+'_label'+str(label1)+'_'+str(label2)+'_attribute'+str(attributeIndex)+'_'+classifierType+'_sigmas = '+str(sigmas)+'mr_and_loocv_result.txt'
    with open(fileName, 'a') as out:
        out.write(str(mr_result) + '\n'  +str(loocv_result) + '\n' )

    #### draw graph
    plt.plot(mr_result, loocv_result, 'ro')
    plt.xlabel('mr_violation_rate')
    plt.ylabel('loocv_result(1=correct,0=misclassifed)')
    # title = ax.set_title("\n".join(wrap("Some really really long long long title I really really need - and just can't - just can't - make it any - simply any - shorter - at all.", 60)))

    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.title("\n".join(wrap('p_value= ' +str(p_value)+', mean = 0, sigmas = '+str(sigmas)+', #instance= '+str(len(mr_result)), 60)))
    plt.savefig('./data_and_results/'+name+'/'+classifierType+'/'+name+'_label'+str(label1)+'_'+str(label2)+'_attribute'+str(attributeIndex)+'_'+classifierType+'_sigmas = '+str(sigmas)+'.png')
   
 

    

