import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv('passengers.csv')
# print(passengers)

# Update sex column to numerical

passengers['Sex'] = passengers['Sex'].map({'male': 0,'female': 1})
#print(passengers)
# Fill the nan values in the age column
# print(passengers['Age'].values)

passengers['Age'].fillna(inplace=True,value= round(passengers['Age'].mean()))

# Create a first class column
passengers['FirstClass'] = passengers['Pclass'].apply(lambda p : 1 if p == 1 else 0)


# Create a second class column

passengers['SecondClass'] = passengers['Pclass'].apply(lambda p : 1 if p == 2 else 0)
#print(passengers)
# Select the desired features
features = passengers[['Sex','Age','FirstClass','SecondClass']]

Survival = passengers['Survived']

# Perform train, test, split

x_train, x_test, y_train, y_test = train_test_split(features,Survival)

# Regularization
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# Create and train the model

model = LogisticRegression()

model.fit(x_train,y_train)


# Score the model on the train data
print(model.score(x_train,y_train))

# Score the model on the test data

print(model.score(x_test,y_test))

# Analyze the coefficients

print(model.coef_)

