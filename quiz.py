import math
#from mnist.best_model_svm_dt import create_split
#from mnist.mnist.utils import model_creation, run_classification_experiment
import sklearn
from sklearn import datasets, svm, metrics
import sys, os
import shutil
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import numpy as np
from joblib import dump, load

sys.path.insert(1, '/home/sonali/mnist')
import utils

#create some data
digits = datasets.load_digits()
X,Y = digits.images,digits.target
data=utils.rescale_resize(X,0.5)

X_train, X_test, y_train, y_test, X_val, y_val=utils.create_split(X,Y,0.30)
best_model_path = "/home/sonali/MLops/mnist/models/s_1_tt_0.25_gamma_0.001/model.joblib"
treeclassifier_model="/home/sonali/MLops/mnist/models/dt_100/model.joblib"



def test_digit_correct_0():
     clf = load(best_model_path)
     images_0=digits.images[digits.target==0]
     eg=np.array(images_0[0]).reshape(1,-1)
     pred=clf.predict(eg)

     print(pred[0])
     assert pred[0]==0
def test_digit_correct_1():
     clf = load(best_model_path)
     images_0=digits.images[digits.target==1]
     eg=np.array(images_0[0]).reshape(1,-1)
     pred=clf.predict(eg)

     print(pred[0])
     assert pred[0]==1

    
def test_digit_correct_2():
     clf = load(best_model_path)
     images_0=digits.images[digits.target==2]
     eg=np.array(images_0[0]).reshape(1,-1)
     pred=clf.predict(eg)

     print(pred[0])
     assert pred[0]==2
    
def test_digit_correct_3():
     clf = load(best_model_path)
     images_0=digits.images[digits.target==3]
     eg=np.array(images_0[0]).reshape(1,-1)
     pred=clf.predict(eg)

     print(pred[0])
     assert pred[0]==3
def test_digit_correct_4():
     clf = load(best_model_path)
     images_0=digits.images[digits.target==4]
     eg=np.array(images_0[0]).reshape(1,-1)
     pred=clf.predict(eg)

     print(pred[0])
     assert pred[0]==4
def test_digit_correct_5():
     clf = load(best_model_path)
     images_0=digits.images[digits.target==5]
     eg=np.array(images_0[0]).reshape(1,-1)
     pred=clf.predict(eg)

     print(pred[0])
     assert pred[0]==5
def test_digit_correct_6():
     clf = load(best_model_path)
     images_0=digits.images[digits.target==6]
     eg=np.array(images_0[0]).reshape(1,-1)
     pred=clf.predict(eg)

     print(pred[0])
     assert pred[0]==6
def test_digit_correct_7():
     clf = load(best_model_path)
     images_0=digits.images[digits.target==7]
     eg=np.array(images_0[0]).reshape(1,-1)
     pred=clf.predict(eg)

     print(pred[0])
     assert pred[0]==7
def test_digit_correct_8():
     clf = load(best_model_path)
     images_0=digits.images[digits.target==8]
     eg=np.array(images_0[0]).reshape(1,-1)
     pred=clf.predict(eg)

     print(pred[0])
     assert pred[0]==8
def test_digit_correct_9():
     clf = load(best_model_path)
     images_0=digits.images[digits.target==9]
     eg=np.array(images_0[0]).reshape(1,-1)
     pred=clf.predict(eg)

     print(pred[0])
     assert pred[0]==9

# for decision tree 

def test_digit_correct_0():
     treeclassifier = load(treeclassifier_model)
     images_0=digits.images[digits.target==0]
     eg=np.array(images_0[0]).reshape(1,-1)
     pred=clf.predict(eg)

     print(pred[0])
     assert pred[0]==0



min_acc_req=0.70
def accuracy_check():
    




