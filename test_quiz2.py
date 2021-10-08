import math
#from mnist.mnist.utils import model_creation, run_classification_experiment
import sklearn
from sklearn import datasets, svm, metrics
import sys, os
import shutil

sys.path.insert(1, '/home/sonali/MLops/mnist/mnist')
import utils

#create some data
digits = datasets.load_digits()
X,Y = digits.images,digits.target
data=utils.rescale_resize(X,0.5)


def test_create_split_100():
    X_train, X_test, y_train, y_test, X_val, y_val=utils.create_split(data[:100],Y[:100],0.5)
    print(X_train.shape[0], X_test.shape[0],X_val.shape[0])
    sum=100
    assert X_train.shape[0]==70 
    assert X_test.shape[0]==20
    assert X_val.shape[0]==10
    assert sum==X_train.shape[0]+X_test.shape[0]+X_val.shape[0]

def test_create_split_10():
    X_train, X_test, y_train, y_test, X_val, y_val=utils.create_split(data[:10],Y[:10],0.5)
    print(X_train.shape[0], X_test.shape[0],X_val.shape[0])
    assert X_train.shape[0]==7
    assert X_test.shape[0]==2
    assert X_val.shape[0]==1
    sum=10
    assert sum==X_train.shape[0]+X_test.shape[0]+X_val.shape[0]
test_create_split_100()
test_create_split_10()
