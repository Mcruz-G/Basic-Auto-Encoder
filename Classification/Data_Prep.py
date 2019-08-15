import numpy as np
import pandas as pd 


def get_data(dataName):
    #Extract data from local file
    dataSet = pd.read_csv('DataSet/'+dataName,sep='\t')

    #Delete some useless columns after data Exploration
    del dataSet['Unnamed: 0']

    return dataSet

def corr_analysis(dataSet,corrCriteria):
    corrmat = dataSet.corr()
    columns = np.full((corrmat.shape[0],), True, dtype=bool)

    for i in range(corrmat.shape[0]):
        for j in range(i+1, corrmat.shape[0]):
            if abs(corrmat.iloc[i,j]) >= corrCriteria:
                if columns[j]:
                    columns[j] = False
                    
    selectedColumns = dataSet.columns[columns]
    dataSet = dataSet[selectedColumns]

    return dataSet



def standardize(X_train):
    from sklearn.preprocessing import StandardScaler

    sc_X = StandardScaler()

    
    X_train = sc_X.fit_transform(X_train.values)
    

    return X_train

