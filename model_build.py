#import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, accuracy_score, r2_score

# #import dataset

df=pd.read_csv(r"C:\Users\Apurv\projects\Taloja\MyApp\advertisement.csv")
print(df.shape)
print(df.head(5))
print(df.columns)

# Fit the data to an ML model
X = df[['TV', 'Radio', 'Newspaper' ]]
y = df['Sales']

model = LinearRegression()
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
model.fit(X, y)
# test_predictions=model.predict(X_test)
# # print("test predictions are : ",test_predictions)
# print(r2_score(y_test, test_predictions))
# print("Model building steps are done")

# Save the model
filename = 'C:\\Users\\Apurv\\projects\\Taloja\\MyApp\\Models\\model.sav'
pickle.dump(model, open(filename, 'wb'))