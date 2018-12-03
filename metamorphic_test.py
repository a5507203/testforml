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

from collections import defaultdict


class Metamorphic_Test:

    def __init__(self, url= './data/iris.data', classifier = 'knn'):
        
        self.df, self.indices, self.uniqueLabels = self.getData(url)
        self.classifier = self.select_classifier(classifier)
        self.result_dic = defaultdict(dict)

    def select_classifier(self, classifier):
        if classifier == 'svm':
            return svm.SVC(kernel='linear', C=1)
        elif classifier == 'knn':
            return KNeighborsClassifier(n_neighbors=5)
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


    def convertDataFormat(self):

        attriutes = self.df.take(self.indices[0:-1], axis=1).values.tolist()
        labels = self.df.take([self.indices[-1]], axis=1).values.flatten()
        return attriutes, labels


    def prepareFormatForTTest(self, df):
        
        return df.take(self.indices[0:-1], axis=1).T.values.tolist()



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

        print("number of attributes: ",len(self.df))
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

        
    def t_test(self,df1,label_i,df2,label_j):

        df1_attributes = self.prepareFormatForTTest(df1)
        df2_attributes = self.prepareFormatForTTest(df2)

        result = stats.ttest_ind(a = df1_attributes,b = df2_attributes, axis = 1 )

        self.result_dic[label_i][label_j] = result




if __name__ == "__main__":



    # # small sample size 
    # name = 'iris'
    # data_url = './data/iris.data'

    # # middium sample size 
    # name = 'car'
    # data_url = './data/car.data'


    # middium sample size 
    name = 'PhishingData'
    data_url = './data/PhishingData.data'


    # # small sample size, binomial classification
    # name = 'Immunotherapy'
    # data_url = './data/Immunotherapy.data'

    # # large sample size, binomial classification
    # name = 'HTRU_2'
    # data_url = './data/HTRU_2.data'


    np.set_printoptions(linewidth = 500)

    # write to file
    sys.stdout = open(name+'_result.txt','wt')

    print('dataset_name: ',name)
    print ('data_url: ',data_url)

    # select classifier
    metamorphic_Test = Metamorphic_Test(url = data_url, classifier = 'knn')
    metamorphic_Test.tTests()
 


