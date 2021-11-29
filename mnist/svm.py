import matplotlib.pyplot as plt
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics,tree
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import DataConversionWarning
from skimage.transform import rescale, resize, downscale_local_mean
import pandas as pd


digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
#5 diff split 
gamma_p = [0.001,0.01,0.1]
kernel=['linear', 'poly', 'rbf']
#list

train_firstrun=[]
dev_firstrun=[]
test_firstrun=[]


def create_split(data, sp):
        X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=sp, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(
        data, digits.target, test_size=sp, shuffle=False)
        #print(np.array(X_train).shape)
        return X_train, X_test, y_train, y_test, X_val, y_val


"""for gamma_p in gamma_p:
        for k in kernel:
            clf = svm.SVC(gamma=gamma_p,kernel=k)
            X_train, X_test, y_train, y_test, X_val, y_val = create_split(data, 0.15)
            clf.fit(X_train,y_train)
            acc_val=clf.score(X_val, y_val)
            acc_train=clf.score(X_train, y_train)
            acc_test=clf.score(X_test, y_test)

            test_firstrun.append(acc_test)
            train_firstrun.append(acc_train)
            dev_firstrun.append(acc_val)
print(train_firstrun)
print(dev_firstrun)
print(test_firstrun)"""

"""
train_secondrun=[]
dev_secondrun=[]
test_secondrun=[]
for g in gamma_p:
        for k in kernel:
            clf = svm.SVC(gamma=g,kernel=k)
            X_train, X_test, y_train, y_test, X_val, y_val = create_split(data, 0.15)
            clf.fit(X_train,y_train)
            acc_val=clf.score(X_val, y_val)
            acc_train=clf.score(X_train, y_train)
            acc_test=clf.score(X_test, y_test)

            train_secondrun.append(acc_test)
            dev_secondrun.append(acc_train)
            test_secondrun.append(acc_val)
print(train_secondrun)
print(dev_secondrun)
print(test_secondrun)"""

train_thirdrun=[]
dev_thirdrun=[]
test_thirdrun=[]
for g in gamma_p:
        for k in kernel:
            clf = svm.SVC(gamma=g,kernel=k)
            X_train, X_test, y_train, y_test, X_val, y_val = create_split(data, 0.15)
            clf.fit(X_train,y_train)
            acc_val=clf.score(X_val, y_val)
            acc_train=clf.score(X_train, y_train)
            acc_test=clf.score(X_test, y_test)

            train_thirdrun.append(acc_test)
            dev_thirdrun.append(acc_train)
            test_thirdrun.append(acc_val)
print(train_thirdrun)
print(dev_thirdrun)
print(test_thirdrun)




  