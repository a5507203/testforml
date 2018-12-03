import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from numbers import Number


def select_classifier(classifier):
    if classifier == 'svm':
        return svm.SVC(kernel='linear', C=1)
    elif classifier == 'knn':
        return KNeighborsClassifier(n_neighbors=5)
    else:
        return None

def getData( url = './data/iris.data'):
    df = pd.read_csv(url)

    #change the value of the attribute to numberical value if is not
    enc = LabelEncoder()

    numberOfColumns = len(df.columns)
    indices = []
    for i in range (numberOfColumns):

        indices.append(i)
        col = df.columns[i]
        if(isinstance(df[col][0], Number)):
            continue
        enc.fit(df[col])
        df[col] = enc.transform(df[col])


    # sperate attributes and labels into lists
    attriutes = df.take(indices[0:-1], axis=1).values.tolist()
    labels = df.take([indices[-1]], axis=1).values.flatten()

    return attriutes, labels



def splitData(n_splits = 10, test_size = 0.1, random_state = 2):
    return ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state = random_state)


if __name__ == "__main__":
    

    # select classifier
    classifier = select_classifier("knn")

    # get data;
    attriutes,labels = getData()

    #split data
    split_strategy = splitData()

    scores = cross_val_score(classifier, attriutes, labels, cv=split_strategy)
    
    #cross validation result
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

