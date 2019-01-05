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
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB
from collections import defaultdict
import os
import errno
from textwrap import wrap

class Metamorphic_Test:

    def __init__(self, url, classifierType, label1, label2):
        
        self.df, self.indices, self.uniqueLabels = self.getData(url, label1, label2)
        self.classifierType = classifierType
        self.classifier = self.select_classifier(classifierType, self.df)
        self.result_dic = defaultdict(dict)
        # self.lOOCV_results = []
        # self.mr_results = []

    def select_classifier(self, classifierType, df):


        attributes, labels = self.convertDataFormat(df)
        if classifierType == 'svm':
            return svm.SVC(C=1.0,gamma= "scale",probability= True).fit(attributes,labels)
        elif classifierType == 'linearSVM':
            return svm.LinearSVC(random_state=0, tol=1e-5).fit(attributes,labels)
        elif classifierType == '3nn':
            return KNeighborsClassifier(n_neighbors=3).fit(attributes,labels)
        elif classifierType == '7nn':
            return KNeighborsClassifier(n_neighbors=7).fit(attributes,labels)
        elif classifierType == '5nn':
            return KNeighborsClassifier(n_neighbors=5).fit(attributes,labels)
        elif classifierType == '10nn':
            return KNeighborsClassifier(n_neighbors=10).fit(attributes,labels)
        elif classifierType == 'gaussianNB':
            return GaussianNB(priors=None, var_smoothing=1e-09).fit(attributes,labels)
        elif classifierType == 'multinomialNB':
            return MultinomialNB(alpha=1.0, class_prior=None).fit(attributes,labels)
        elif classifierType == 'complementNB':
            return ComplementNB().fit(attributes,labels)        
        else:
            return None


    def getData(self, url,label1, label2 ):
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

        
        df1 = df[df['class'] == label1].copy()
        df2 = df[df['class'] == label2].copy()

        for un in uniqueLabels:
            print(len(df[df['class'] == un]), ' ', un)
        df3 = pd.concat([df1,df2])
        for i in range(len(df3.columns)-1):
            col = df3.columns[i]
            df3[col] = (df3[col] - df3[col].mean())/df3[col].std()
        
        # print(df3)
        return df3, indices, uniqueLabels


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


    def cv(self, label2):
        predictions = []
        classifier = self.select_classifier(self.classifierType,self.trainingDf)
        testAttr, testLabel = self.convertDataFormat(self.testDf)
        count = 0

        results = self.predict(classifier, testAttr)
        print(results)
        for prediction in results:
            if prediction != label2:
                count += 1
                predictions.append(0)
            else:
                predictions.append(1)
        return predictions,count
            # print(b)

    def getVariance(self, df, attributeIndex):
        return df.loc[:,df.columns[attributeIndex]].std()

    def metamorphic_Test_continious_attr(self, label1, label2, attributeIndex, mu = 0, sigmas = [0.2,0.5,1,2], repeatForEverySigma = 20 ):

        df1 = self.getAllInstanceBylabel(label1).copy()
        df2 = self.getAllInstanceBylabel(label2).copy()


        std = self.getVariance(self.df,attributeIndex)

        # random 50 samples as test samples
        # testIndices = np.random.randint(low=1, high=len(df2)-1, size=int(len(df2)*0.2))
    
        testIndices = np.random.choice(range(len(df2)), int(len(df2)*0.2), replace=False)
        # print(testIndices)
        # print(df2)
        self.testDf = df2.iloc[testIndices].copy()
        # print(self.testDf)
        # leave others as training set for CV
        self.trainingDf = self.df.drop(df2.index[testIndices])
        # print(self.trainingDf)
        for uniqueLabel in self.uniqueLabels:
            print(len(self.getAllInstanceBylabel(uniqueLabel).copy()),' label ' ,uniqueLabel)

      
        repeatTime = len(sigmas)*repeatForEverySigma
        ori_attri, ori_label = self.convertDataFormat(self.testDf)
        source_predictions = self.predict(self.classifier,ori_attri)       
        # print (source_prediction)      
        i = 0
        followup_predictions = np.empty(shape = [len(self.testDf),repeatTime], dtype=int)
        for sigma in sigmas:
            for j in range(repeatForEverySigma):
                df3 = self.testDf.copy()
              
                df3[df3.columns[attributeIndex]] += np.random.normal(mu, sigma*std, len(df3))
                attri, label = self.convertDataFormat(df3)
                prediction = self.predict(self.classifier,attri)
                followup_predictions[:,i] = prediction
                i+=1

        results = []
        count = 0
        i=0
        for prediction in followup_predictions:
            source_prediction = source_predictions[i]
            result = ((prediction == source_prediction).sum())
            error = 1-float(result)/float(repeatTime)
            if(error >= 0.5):
                count+=1
            results.append(error)
            i+=1
        return results,count,std
      

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

def experiment(name,label1,label2,repeatForEverySigma,attributeIndex,p_value,sigmas, classifierTypes = ['5nn','10nn','gaussianNB','svm']): #'gaussianNB',

    p_value = str('{:.8e}'.format(p_value))
    data_url = './data_and_results_ovr/'+name+'/'+name+'.data'

    for classifierType in classifierTypes:

        print('dataset_name: ',name)
        print ('data_url: ',data_url)
        print ('classifierType: ',classifierType)

        figurePath = './data_and_results_ovr/'+name+'/'+p_value+'/'+classifierType
        
        createDictionary(figurePath)

        metamorphic_Test = Metamorphic_Test(url = data_url, classifierType = classifierType, label1 = label1, label2 = label2 )
      
        mr_result,larger_violation_number,std = metamorphic_Test.metamorphic_Test_continious_attr(label1 = label1, label2 = label2, attributeIndex = attributeIndex, mu = 0, sigmas=sigmas,repeatForEverySigma=repeatForEverySigma)
        
        loocv_result,error = metamorphic_Test.cv(label2=label2)

        confidence = 0
        #histogram = []
        for i in range( len(mr_result)):
            if(mr_result[i] >= 0.5 and loocv_result[i] == 0):
                confidence+=1

            # if(loocv_result[i] == 0):
            #     histogram.append(mr_result[i])

        confidence = float(confidence)/float(larger_violation_number+0.000000000000000000001)
        #### write test result to file
  
        fileName = figurePath +'/'+name+'_label'+str(label1)+'_'+str(label2)+'_attribute'+str(attributeIndex)+'_'+classifierType+'_sigmas = '+str(sigmas)+'_std = '+str(std) +'_mr_and_loocv_result.txt'
        with open(fileName, 'w+') as out:
            out.write(str(mr_result) + '\n'  +str(loocv_result) + '\n' )

        #### draw graph
        plt.plot(mr_result, loocv_result, 'ro')
        plt.xlabel('mr_violation_rate')
        plt.ylabel('loocv_result (1=correct, 0=misclassifed)')
        plt.tight_layout()
        plt.subplots_adjust(top=0.8)
        instanceNumber = len(mr_result)
        # plt.hist(histogram)
        plt.title("\n".join(wrap('p_value= ' +p_value+', mean = 0, sigmas = '+str(sigmas)+', #instance= '+str(instanceNumber)+','+'\n'+'(MRVRate>0.5 and misclassified)/(MRVRate>0.5) = '+str('{:.3e}'.format(confidence)) +', error_rate= '+str('{:.3e}'.format(float(error)/float(instanceNumber))), 60)))
        plt.savefig(figurePath+'/'+name+'_label'+str(label1)+'_'+str(label2)+'_attribute'+str(attributeIndex)+'_'+classifierType+'_sigmas = '+str(sigmas)+'.png')
        plt.clf()
        plt.cla()
        plt.close()
        

np.set_printoptions(linewidth = 500)

experiment(name = 'avila-tr', label1=1,label2=2,attributeIndex=0,sigmas=[0.1],p_value=2.95635742e-001,repeatForEverySigma = 500)
experiment(name = 'avila-tr', label1=1,label2=2,attributeIndex=0,sigmas=[0.2],p_value=2.95635742e-001,repeatForEverySigma = 500)
experiment(name = 'avila-tr', label1=1,label2=2,attributeIndex=0,sigmas=[0.5],p_value=2.95635742e-001,repeatForEverySigma = 500)
experiment(name = 'avila-tr', label1=1,label2=2,attributeIndex=0,sigmas=[1],p_value=2.95635742e-001,repeatForEverySigma = 500)
experiment(name = 'avila-tr', label1=1,label2=2,attributeIndex=0,sigmas=[2],p_value=2.95635742e-001,repeatForEverySigma = 500)
experiment(name = 'avila-tr', label1=1,label2=2,attributeIndex=0,sigmas=[3],p_value=2.95635742e-001,repeatForEverySigma = 500)
experiment(name = 'avila-tr', label1=1,label2=2,attributeIndex=0,sigmas=[4],p_value=2.95635742e-001,repeatForEverySigma = 500)
experiment(name = 'avila-tr', label1=1,label2=2,attributeIndex=0,sigmas=[5],p_value=2.95635742e-001,repeatForEverySigma = 500)
experiment(name = 'avila-tr', label1=1,label2=2,attributeIndex=0,sigmas=[6],p_value=2.95635742e-001,repeatForEverySigma = 500)


# experiment(name = 'avila-tr', label1=1,label2=2,attributeIndex=8,sigmas=[0.1],p_value=9.20291865e-138,repeatForEverySigma = 500)
# experiment(name = 'avila-tr', label1=1,label2=2,attributeIndex=8,sigmas=[0.2],p_value=9.20291865e-138,repeatForEverySigma = 500)
# experiment(name = 'avila-tr', label1=1,label2=2,attributeIndex=8,sigmas=[0.5],p_value=9.20291865e-138,repeatForEverySigma = 500)
# experiment(name = 'avila-tr', label1=1,label2=2,attributeIndex=8,sigmas=[1],p_value=9.20291865e-138,repeatForEverySigma = 500)
# experiment(name = 'avila-tr', label1=1,label2=2,attributeIndex=8,sigmas=[2],p_value=9.20291865e-138,repeatForEverySigma = 500)
# experiment(name = 'avila-tr', label1=1,label2=2,attributeIndex=8,sigmas=[3],p_value=9.20291865e-138,repeatForEverySigma = 500)
# experiment(name = 'avila-tr', label1=1,label2=2,attributeIndex=8,sigmas=[4],p_value=9.20291865e-138,repeatForEverySigma = 500)
# experiment(name = 'avila-tr', label1=1,label2=2,attributeIndex=8,sigmas=[5],p_value=9.20291865e-138,repeatForEverySigma = 500)
# experiment(name = 'avila-tr', label1=1,label2=2,attributeIndex=8,sigmas=[6],p_value=9.20291865e-138,repeatForEverySigma = 500)




experiment(name = 'data_banknote_authentication', label1=1,label2=2,attributeIndex=3,sigmas=[0.1],p_value=3.85967572e-001,repeatForEverySigma = 500)
experiment(name = 'data_banknote_authentication', label1=1,label2=2,attributeIndex=3,sigmas=[0.2],p_value=3.85967572e-001,repeatForEverySigma = 500)
experiment(name = 'data_banknote_authentication', label1=1,label2=2,attributeIndex=3,sigmas=[0.5],p_value=3.85967572e-001,repeatForEverySigma = 500)
experiment(name = 'data_banknote_authentication', label1=1,label2=2,attributeIndex=3,sigmas=[1],p_value=3.85967572e-001,repeatForEverySigma = 500)
experiment(name = 'data_banknote_authentication', label1=1,label2=2,attributeIndex=3,sigmas=[2],p_value=3.85967572e-001,repeatForEverySigma = 500)
experiment(name = 'data_banknote_authentication', label1=1,label2=2,attributeIndex=3,sigmas=[3],p_value=3.85967572e-001,repeatForEverySigma = 500)
experiment(name = 'data_banknote_authentication', label1=1,label2=2,attributeIndex=3,sigmas=[4],p_value=3.85967572e-001,repeatForEverySigma = 500)
experiment(name = 'data_banknote_authentication', label1=1,label2=2,attributeIndex=3,sigmas=[5],p_value=3.85967572e-001,repeatForEverySigma = 500)
experiment(name = 'data_banknote_authentication', label1=1,label2=2,attributeIndex=3,sigmas=[6],p_value=3.85967572e-001,repeatForEverySigma = 500)

# experiment(name = 'data_banknote_authentication', label1=1,label2=2,attributeIndex=0,sigmas=[0.1],p_value=5.74096537e-224,repeatForEverySigma = 500)
# experiment(name = 'data_banknote_authentication', label1=1,label2=2,attributeIndex=0,sigmas=[0.2],p_value=5.74096537e-224,repeatForEverySigma = 500)
# experiment(name = 'data_banknote_authentication', label1=1,label2=2,attributeIndex=0,sigmas=[0.5],p_value=5.74096537e-224,repeatForEverySigma = 500)
# experiment(name = 'data_banknote_authentication', label1=1,label2=2,attributeIndex=0,sigmas=[1],p_value=5.74096537e-224,repeatForEverySigma = 500)
# experiment(name = 'data_banknote_authentication', label1=1,label2=2,attributeIndex=0,sigmas=[2],p_value=5.74096537e-224,repeatForEverySigma = 500)
# experiment(name = 'data_banknote_authentication', label1=1,label2=2,attributeIndex=0,sigmas=[3],p_value=5.74096537e-224,repeatForEverySigma = 500)
# experiment(name = 'data_banknote_authentication', label1=1,label2=2,attributeIndex=0,sigmas=[4],p_value=5.74096537e-224,repeatForEverySigma = 500)
# experiment(name = 'data_banknote_authentication', label1=1,label2=2,attributeIndex=0,sigmas=[5],p_value=5.74096537e-224,repeatForEverySigma = 500)
# experiment(name = 'data_banknote_authentication', label1=1,label2=2,attributeIndex=0,sigmas=[6],p_value=5.74096537e-224,repeatForEverySigma = 500)




experiment(name = 'sensor_readings_24', label1=1,label2=2,attributeIndex=15,sigmas=[0.2],p_value=9.27828066e-001,repeatForEverySigma = 500)
experiment(name = 'sensor_readings_24', label1=1,label2=2,attributeIndex=15,sigmas=[0.5],p_value=9.27828066e-001,repeatForEverySigma = 500)
experiment(name = 'sensor_readings_24', label1=1,label2=2,attributeIndex=15,sigmas=[1],p_value=9.27828066e-001,repeatForEverySigma = 500)
experiment(name = 'sensor_readings_24', label1=1,label2=2,attributeIndex=15,sigmas=[2],p_value=9.27828066e-001,repeatForEverySigma = 500)
experiment(name = 'sensor_readings_24', label1=1,label2=2,attributeIndex=15,sigmas=[3],p_value=9.27828066e-001,repeatForEverySigma = 500)
experiment(name = 'sensor_readings_24', label1=1,label2=2,attributeIndex=15,sigmas=[4],p_value=9.27828066e-001,repeatForEverySigma = 500)
experiment(name = 'sensor_readings_24', label1=1,label2=2,attributeIndex=15,sigmas=[5],p_value=9.27828066e-001,repeatForEverySigma = 500)
experiment(name = 'sensor_readings_24', label1=1,label2=2,attributeIndex=15,sigmas=[6],p_value=9.27828066e-001,repeatForEverySigma = 500)

# experiment(name = 'sensor_readings_24', label1=1,label2=2,attributeIndex=12,sigmas=[0.2],p_value=1.55853603e-064,repeatForEverySigma = 500)
# experiment(name = 'sensor_readings_24', label1=1,label2=2,attributeIndex=12,sigmas=[0.5],p_value=1.55853603e-064,repeatForEverySigma = 500)
# experiment(name = 'sensor_readings_24', label1=1,label2=2,attributeIndex=12,sigmas=[1],p_value=1.55853603e-064,repeatForEverySigma = 500)
# experiment(name = 'sensor_readings_24', label1=1,label2=2,attributeIndex=12,sigmas=[2],p_value=1.55853603e-064,repeatForEverySigma = 500)
# experiment(name = 'sensor_readings_24', label1=1,label2=2,attributeIndex=12,sigmas=[3],p_value=1.55853603e-064,repeatForEverySigma = 500)
# experiment(name = 'sensor_readings_24', label1=1,label2=2,attributeIndex=12,sigmas=[4],p_value=1.55853603e-064,repeatForEverySigma = 500)
# experiment(name = 'sensor_readings_24', label1=1,label2=2,attributeIndex=12,sigmas=[5],p_value=1.55853603e-064,repeatForEverySigma = 500)
# experiment(name = 'sensor_readings_24', label1=1,label2=2,attributeIndex=12,sigmas=[6],p_value=1.55853603e-064,repeatForEverySigma = 500)





experiment(name = 'Frogs_MFCCs', label1=1,label2=2,attributeIndex=11,sigmas=[0.1],p_value=9.18645333e-003,repeatForEverySigma = 500)
experiment(name = 'Frogs_MFCCs', label1=1,label2=2,attributeIndex=11,sigmas=[0.2],p_value=9.18645333e-003,repeatForEverySigma = 500)
experiment(name = 'Frogs_MFCCs', label1=1,label2=2,attributeIndex=11,sigmas=[0.5],p_value=9.18645333e-003,repeatForEverySigma = 500)
experiment(name = 'Frogs_MFCCs', label1=1,label2=2,attributeIndex=11,sigmas=[1],p_value=9.18645333e-003,repeatForEverySigma = 500)
experiment(name = 'Frogs_MFCCs', label1=1,label2=2,attributeIndex=11,sigmas=[2],p_value=9.18645333e-003,repeatForEverySigma = 500)
experiment(name = 'Frogs_MFCCs', label1=1,label2=2,attributeIndex=11,sigmas=[3],p_value=9.18645333e-003,repeatForEverySigma = 500)
experiment(name = 'Frogs_MFCCs', label1=1,label2=2,attributeIndex=11,sigmas=[4],p_value=9.18645333e-003,repeatForEverySigma = 500)
experiment(name = 'Frogs_MFCCs', label1=1,label2=2,attributeIndex=11,sigmas=[5],p_value=9.18645333e-003,repeatForEverySigma = 500)
experiment(name = 'Frogs_MFCCs', label1=1,label2=2,attributeIndex=11,sigmas=[6],p_value=9.18645333e-003,repeatForEverySigma = 500)

# experiment(name = 'Frogs_MFCCs', label1=1,label2=2,attributeIndex=5,sigmas=[0.1],p_value=0,repeatForEverySigma = 500)
# experiment(name = 'Frogs_MFCCs', label1=1,label2=2,attributeIndex=5,sigmas=[0.2],p_value=0,repeatForEverySigma = 500)
# experiment(name = 'Frogs_MFCCs', label1=1,label2=2,attributeIndex=5,sigmas=[0.5],p_value=0,repeatForEverySigma = 500)
# experiment(name = 'Frogs_MFCCs', label1=1,label2=2,attributeIndex=5,sigmas=[1],p_value=0,repeatForEverySigma = 500)
# experiment(name = 'Frogs_MFCCs', label1=1,label2=2,attributeIndex=5,sigmas=[2],p_value=0,repeatForEverySigma = 500)
# experiment(name = 'Frogs_MFCCs', label1=1,label2=2,attributeIndex=5,sigmas=[3],p_value=0,repeatForEverySigma = 500)
# experiment(name = 'Frogs_MFCCs', label1=1,label2=2,attributeIndex=5,sigmas=[4],p_value=0,repeatForEverySigma = 500)
# experiment(name = 'Frogs_MFCCs', label1=1,label2=2,attributeIndex=5,sigmas=[5],p_value=0,repeatForEverySigma = 500)
# experiment(name = 'Frogs_MFCCs', label1=1,label2=2,attributeIndex=5,sigmas=[6],p_value=0,repeatForEverySigma = 500)
