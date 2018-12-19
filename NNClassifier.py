import pandas as pd
import tensorflow as tf

TRAINING_FILE = ""
train = pd.read_csv(TRAINING_FILE)
y_train, X_train = train['label'], train[['feature1', 'feature2']].fillna(0)



def getData( url ):
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