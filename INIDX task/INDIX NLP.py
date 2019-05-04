# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
 
 
train_data = pd.read_csv('bmv_test_set.csv')
 
 
# Cleaning the Train Data Texts
train_attr_list = []
for row in range(len(train_data)):
    addl_Attr_train = re.sub('[^a-zA-Z]', ' ', train_data['additionalAttributes'][row])
    addl_Attr_train = addl_Attr_train.lower()
    addl_Attr_train = addl_Attr_train.split()
# Stemming   
    ps = PorterStemmer()
    addl_Attr_train = [ps.stem(word) for word in addl_Attr_train if not word in set(stopwords.words('english'))]
    addl_Attr_train = ' '.join(addl_Attr_train)
    train_attr_list.append(addl_Attr_train)
#
CV_train = CountVectorizer(max_features = 500)
X_train = CV_train.fit_transform(train_attr_list).toarray()
 
cluster_inertia = []
for i in range(1,21):
    model = KMeans(n_clusters = i,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)
    model.fit(X_train)
    cluster_inertia.append(model.inertia_)
 
print(model)
plt.plot(range(1,21),cluster_inertia)
plt.title('The Elbow Method')
plt.xlabel('clusters')
plt.ylabel('inertia')
plt.show()
 
model = KMeans(n_clusters = 16,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)
Y_train = model.fit_predict(X_train)
print(Y_train)
 
 

#Finding train accuracy using K-Fold
RFR = RandomForestRegressor(n_estimators = 300,random_state = 0)
kfold = KFold(n_splits=10, random_state=0)
train_accuracies = cross_val_score(estimator = RFR,X = X_train,y = Y_train,cv = kfold)
print("Train Accuracies = ", train_accuracies)
print("Train Data Accuracies SD = ", train_accuracies.std())
print("Train Data Accuracies mean = ", train_accuracies.mean())
 
 
 
 
#######################################################################################################
#Reading test data
test_data = pd.read_csv('BMV2.csv')
 
 
# Cleaning the Test Data Texts
test_attr_list = []
for row in range(len(test_data)):
    addl_Attr_test = re.sub('[^a-zA-Z]', ' ', test_data['additionalAttributes'][row])
    addl_Attr_test = addl_Attr_test.lower()
    addl_Attr_test = addl_Attr_test.split()
# Stemming   
    ps = PorterStemmer()
    addl_Attr_test = [ps.stem(word) for word in addl_Attr_test if not word in set(stopwords.words('english'))]
    addl_Attr_test = ' '.join(addl_Attr_test)
    test_attr_list.append(addl_Attr_test)
#
CV_test = CountVectorizer(max_features = 500)
X_test = CV_test.fit_transform(test_attr_list).toarray()
 
 
#building the model to predit Y_test.
RFR = RandomForestRegressor(n_estimators = 300,random_state = 0)
RFR.fit(X_train,Y_train)
Y_test = RFR.predict(X_test)
 
#Finding accuracy using K-Fold
kfold = KFold(n_splits=10, random_state=0)
test_accuracies = cross_val_score(estimator = RFR,X = X_test,y = Y_test,cv = kfold)
print("Test Accuracies = ", test_accuracies)
print("Test Data Accuracies SD = ", test_accuracies.std())
print("Test Data Accuracies mean = ", test_accuracies.mean())
 
 
 
new_Y_train = np.reshape(Y_train,(20000,1))
new_id = train_data.iloc[:,:1].values
new_id = np.append(arr=new_id, values = new_Y_train,axis = 1)
 
 
plt.scatter(new_id[Y_train == 0,0],Y_train[Y_train ==0,1],s = 20,c = 'red',label = 'C1')
'''
plt.scatter(new_id[Y_train == 1,0],Y_train[Y_train ==1,0],s = 100,c = 'magenta',label = 'C2')
plt.scatter(new_id[Y_train == 2,0],Y_train[Y_train ==2,0],s = 210,c = 'green',label = 'C3')
plt.scatter(new_id[Y_train == 3,0],Y_train[Y_train ==3,0],s = 320,c = 'blue',label = 'C4')
 

plt.scatter(X[Y_kmeans == 4,0],X[Y_kmeans ==4,1],s = 20,c = 'yellow',label = 'C5')
plt.scatter(X[Y_kmeans == 5,0],X[Y_kmeans ==5,1],s = 20,c = 'cyan',label = 'C6')
plt.scatter(X[Y_kmeans == 6,0],X[Y_kmeans ==6,1],s = 20,c = 'orange',label = 'C7')
plt.scatter(X[Y_kmeans == 7,0],X[Y_kmeans ==7,1],s = 20,c = 'grey',label = 'C8')
plt.scatter(X[Y_kmeans == 8,0],X[Y_kmeans ==8,1],s = 20,c = 'peru',label = 'C9')
plt.scatter(X[Y_kmeans == 9,0],X[Y_kmeans ==9,1],s = 20,c = 'olive',label = 'C10')
plt.scatter(X[Y_kmeans == 10,0],X[Y_kmeans ==10,1],s = 20,c = 'purple',label = 'C11')
plt.scatter(X[Y_kmeans == 11,0],X[Y_kmeans ==11,1],s = 20,c = 'dodgerblue',label = 'C12')
plt.scatter(X[Y_kmeans == 12,0],X[Y_kmeans ==12,1],s = 20,c = 'violet',label = 'C13')
plt.scatter(X[Y_kmeans == 13,0],X[Y_kmeans ==13,1],s = 20,c = 'hotpink',label = 'C14')
plt.scatter(X[Y_kmeans == 14,0],X[Y_kmeans ==14,1],s = 20,c = 'cornsilk',label = 'C15')
plt.scatter(X[Y_kmeans == 15,0],X[Y_kmeans ==15,1],s = 20,c = 'greenyellow',label = 'C16')
'''
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],s = 100,c = 'black',label = 'CENTRIOID')
plt.title('CATEGERIOUS')
plt.xlabel('INDEX')
plt.ylabel('TYPE OF RECORDS')
plt.legend()
plt.show()
 
 


