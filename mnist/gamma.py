import matplotlib.pyplot as plt
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
digits = datasets.load_digits()
n_samples = len(digits.images)
   
data = digits.images.reshape((n_samples, -1))
X,Y = digits.images,digits.target 
   



#accurcy list
a=[]
def testf(digits, params):
   
    clf = svm.SVC(gamma=params)
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.15, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(
        data, digits.target, test_size=0.15, shuffle=False)
   
    clf.fit(X_train, y_train.reshape(-1,1))

    acc = clf.score(X_val,y_val)
    a.append(acc)
    print("gamma-val_accuracy ",params,"--->",acc)

#gamman values list
gamma=[0.000001, 0.00001,0.0001,0.001,0.01,0.1,0.2,1]
for i in gamma:
    testf(digits,i)
#finding best gamma values
max_accuracy=max(a)
#print(max_accuracy)
optimized_gamma=gamma[a.index(max_accuracy)]
print("Best gamma value", optimized_gamma)

def acc(digits, params):
    clf = svm.SVC(gamma=params)
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.15, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(
        data, digits.target, test_size=0.15, shuffle=False)
   
    clf.fit(X_train, y_train.reshape(-1,1))
    print("Test Accuracy --->",clf.score(X_train,y_train))
    print("Validation Accuracy --->",clf.score(X_val,y_val))
    print("Test Accuracy --->",clf.score(X_test,y_test))
acc(digits,optimized_gamma)



