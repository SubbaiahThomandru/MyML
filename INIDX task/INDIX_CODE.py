import pandas as pd
import numpy as np
import re
import nltk
from sklearn.preprocessing import LabelEncoder
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
#
###########################################################
#     Model Training code.
###########################################################
##Reading the train data file#
train_data = pd.read_csv('training_set.csv')
#
#>>> Converting text to number
Y_train = train_data.iloc[:,2:].values
labelencoder = LabelEncoder()
Y_train[:,0] = labelencoder.fit_transform(Y_train[:,0])
Y_train = pd.DataFrame(Y_train)
Y_train = Y_train.values
Y_train = Y_train.reshape(len(Y_train)).astype(int)
#<<<
#>>> Preprocessing the data
# Cleaning the train Data Texts
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
#Creating bag of words
CV_train = CountVectorizer(max_features = 500)
#<<<
#building sparse matrix on train data
X_train  = CV_train.fit_transform(train_attr_list).toarray()
#
#>>>MODEL SELECTION 
#Data splitting 
X1,X2,Y1,Y2 = train_test_split(X_train,Y_train,test_size = 0.25,random_state = 0)
#
Y1 = Y1.values
Y1 = Y1.reshape(len(Y1)).astype(int)
Y2 = Y2.values
Y2 = Y2.reshape(len(Y2)).astype(int)
RFC = RandomForestClassifier(n_estimators = 300,criterion = 'entropy',random_state = 0)
RFC.fit(X1,Y1)
#
Y2_predit = RFC.predict(X2)
#
CM = confusion_matrix(Y2,Y2_predit)
print(CM)
#
kfold = KFold(n_splits=10, random_state=0)
train_accuracies = cross_val_score(estimator = RFC,X = X2,y = Y2,cv = kfold)
print("Train Accuracies = ", train_accuracies)
print("Train Data Accuracies SD = ", train_accuracies.std())
print("Train Data Accuracies mean = ", train_accuracies.mean())
#<<<
###################################################################################################
#     Model Prediction code.
###################################################################################################
#Reading the Test data file
test_data = pd.read_csv('testing_set.csv')

#>>> Preprocessing the data, i.e. cleaning the texts.
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
#Creating bag of words
CV_test = CountVectorizer(max_features = 500)
#<<<`
#building sparse matrix on test data
X_test = CV_test.fit_transform(test_attr_list).toarray()
#
#>>>Predicting the Y values in the test data, model building
RFC = RandomForestClassifier(n_estimators = 300,criterion = 'entropy',random_state = 0)
RFC.fit(X_train,Y_train)
Y_test = RFC.predict(X_test)
#<<<
#Finding Accuracies
kfold = KFold(n_splits=10, random_state=0)
test_accuracies = cross_val_score(estimator = RFC,X = X_test,y = Y_test,cv = kfold)
print("Test Accuracies = ", test_accuracies)
print("Test Data Accuracies SD = ", test_accuracies.std())
print("Test Data Accuracies mean = ", test_accuracies.mean())
#
#>>> Coverting numeric to labels
Y_test = labelencoder.inverse_transform(Y_test)
#
#>>> adding the predicted values to the test data as label column
Y_test = np.reshape(Y_test,(len(Y_test),1))
test_data1 = np.append(arr=test_data, values = Y_test,axis = 1)
test_data1 = pd.DataFrame(test_data1,columns = ["id", "additionalAttributes", "labels"])
#<<<
#>>>Writeing Testing set to submissions file
with open('submissions.csv','w') as outfile:
    test_data1.to_csv(outfile, index=False)
#<<<