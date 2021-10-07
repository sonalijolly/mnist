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

X_train, X_test, y_train, y_test, X_val, y_val=utils.create_split(data[:100],Y[:100],0.25)

parent_dir='/home/sonali/MLops/mnist/models_test'
dir='gamma_0.01'

def test_model_writing():
    

    model_path,clf=utils.model_creation(X_train,y_train)
    assert os.path.isfile(model_path+"/model.joblib")

def test_small_data_overfit_checking():
    acc_train,f1_train,acc_test,f1_test=utils.run_classification_experiment(X_train,y_train,X_val,y_val)
    #for training
    print("Train metrics")
    print("Train accuarcy : ",acc_train)
    print("Test accuarcy : ",acc_test)


    assert acc_train>0.95
    assert f1_train>0.95
"""test_model_writing()
test_small_data_overfit_checking()
"""







    




    
    #run_classification_experiment(data, expeted-model-file)

    #assert os.path.isfile(expected-model-file)

test_model_writing()