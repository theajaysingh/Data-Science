import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

#print(cancer.DESCR) # Print the data set description

cancer.keys()


#question 0

def answer_zero():
    return len(cancer['feature_names'])
    
answer_zero()

#Question 1

def answer_one():
    
    # Your code here
    
    columns = np.append(cancer.feature_names, 'target')
    
    index = pd.RangeIndex(start = 0, stop = 569, step = 1)
    
    data = np.column_stack((cancer.data, cancer.target))
    
    df = pd.DataFrame(data = data , index = index, columns = columns)
    return df


answer_one()

#Question 2

def answer_two():
    cancerdf = answer_one()
    
    index = ['malignant','benign']
    
    malignant= cancerdf[cancerdf.target == 0]
    
    benign = cancerdf[cancerdf.target == 1.0]
    
    
    data = (len(malignant),len(benign))
    
    return pd.Series(data,index = index)

answer_two()

#Question 3

def answer_three():
    cancerdf = answer_one()
    
    X = cancerdf.drop('target',axis=1)
    
    y = cancerdf.get('target')
    
    return X, y

#Question 4

from sklearn.model_selection import train_test_split

def answer_four():
    X, y = answer_three()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    
    # Your code here
    
    return X_train, X_test, y_train, y_test
    
#Question 5

from sklearn.neighbors import KNeighborsClassifier

def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    
    knn = KNeighborsClassifier(n_neighbors = 1)
    
    knn.fit(X_train, y_train)
    
    knn.score(X_test, y_test)
    
    return knn
    
#Question 6

def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    
    # Your code here
    knn = answer_five()
    
    return knn.predcict(means)

#Question 7

def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    
    # Your code here
    predict = knn.predict(x_train)
    
    return predict
    
    
#Question 8

def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    
    score = knn.score(X_test, y_test)
    
    return score

answer_eight()
