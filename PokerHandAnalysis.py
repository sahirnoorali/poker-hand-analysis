# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 14:31:40 2017

Authors: K132047 | Sahir
           
Data Science Project

Code:   Presents a Comparision of Different Classifiers and
        Applies Multi-Layer Perceptron Classifier on the UCI
        Poker Hand Data Set
"""
#-------------------------------------------------------------------------
# All the Libraries: 
#------------------------------------------------------------------------- 
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OutputCodeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
#----------------------------------------------------------------
#Read the Training and Testing Data:
#----------------------------------------------------------------
data_train = pd.read_csv(filepath_or_buffer="poker-hand-training-true.data", sep=',', header=None)
data_test = pd.read_csv(filepath_or_buffer="poker-hand-testing.data", sep=',', header=None)
#----------------------------------------------------------------
#Print it's Shape to get an idea of the data set:
#----------------------------------------------------------------
print(data_train.shape)
print(data_test.shape)
#----------------------------------------------------------------
#Prepare the Data for Training and Testing:
#----------------------------------------------------------------
#Ready the Train Data
array_train = data_train.values
data_train = array_train[:,0:10]
label_train = array_train[:,10]
#Ready the Test Data
array_test = data_test.values
data_test = array_test[:,0:10]
label_test = array_test[:,10]
#----------------------------------------------------------------
# Scaling the Data for our Main Model
#----------------------------------------------------------------
# Scale the Data to Make the NN easier to converge
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(data_train)  
# Transform the training and testing data
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)
#----------------------------------------------------------------
#Apply the MLPClassifier:
#----------------------------------------------------------------
acc_array = [0] * 5
for s in range (1,6):
    #Init MLPClassifier
    clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(64,64),
                        activation='tanh', learning_rate_init=0.02,max_iter=2000,random_state=s)
    #Fit the Model
    result = clf.fit(data_train, label_train)
    #Predict
    prediction = clf.predict(data_test)
    #Get Accuracy
    acc = accuracy_score(label_test, prediction)
    #Store in the Array
    acc_array[s-1] = acc
#----------------------------------------------------------------
#Fetch & Print the Results:
#----------------------------------------------------------------
    print(classification_report(label_test,prediction))
    print("Accuracy using MLPClassifier and Random Seed:",s,":",str(acc)) 
    print(confusion_matrix(label_test, prediction))
print("Mean Accuracy using MLPClassifier Classifier: ",np.array(acc_array).mean())
#----------------------------------------------------------------
# Init the Models for Comparision
#----------------------------------------------------------------
models = [BaggingClassifier(), RandomForestClassifier(), AdaBoostClassifier(), 
          KNeighborsClassifier(),GaussianNB(),tree.DecisionTreeClassifier(),
          svm.SVC(kernel='linear', C=1), OutputCodeClassifier(BaggingClassifier()),
            OneVsRestClassifier(svm.SVC(kernel='linear'))]

model_names = ["Bagging with DT", "Random Forest", "AdaBoost", "KNN","Naive Bayes","Decision Tree",
               "Linear SVM","OutputCodeClassifier with Linear SVM" ,"OneVsRestClassifier with Linear SVM"]
#----------------------------------------------------------------
# Run Each Model
#----------------------------------------------------------------
for model,name in zip(models,model_names):
    model.fit(data_train, label_train)
    # Display the relative importance of each attribute
    if name == "Random Forest":
        print(model.feature_importances_)   
    #Predict
    prediction = model.predict(data_test)
    # Print Accuracy
    acc = accuracy_score(label_test, prediction)
    print("Accuracy Using",name,": " + str(acc)+'\n')
    print(classification_report(label_test,prediction))
    print(confusion_matrix(label_test, prediction))



