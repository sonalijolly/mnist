import math
#from mnist.mnist.utils import model_creation, run_classification_experiment
import sklearn
from numpy.lib.function_base import average
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
import numpy as np
import os
from joblib import dump, load


digits = datasets.load_digits()

def create_split(data,y, sp):
        X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=0.30, shuffle=False)
        
        X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.66, shuffle=False)
        
        return X_train, X_test, y_train, y_test, X_val, y_val
        
        

def rescale_resize(data,shape):
    data_preprocessed=[]
    data_processed=[]
    for image in data:
        image_rescaled = rescale(image, shape, anti_aliasing=False)
        data_preprocessed.append(image_rescaled)
    data_processed = np.asarray(data_preprocessed)
    data_processed = data_processed.reshape((len(data), -1))
    return data_processed

def classfreport(model,y_test,preds):
    print(f"Classification report for classifier {model}:\n"
      f"{metrics.classification_report(y_test, preds)}\n")
def test(clf,X,y):
    pred=clf.predict(X)
    acc=metrics.accuracy_score(y_pred=pred,y_true=y)
    f1=metrics.f1_score(y_pred=pred,y_true=y,average="macro")
    
    return acc,f1

def get_random_acc(y):
    return max(np.bincount(y))/len(y)   
'''my_dir='/home/sonali/MLops/mnist/models_test'

if os.path.exists(my_dir):
    shutil.rmtree(my_dir) '''

def run_classification_experiment(X_train,y_train,X_val,y_val):
    gamma=0.01
    #1000 iterations for overfitting
    clf=svm.SVC(gamma=gamma,max_iter=1000)
    clf.fit(X_train,y_train)
    acc_train,f1_train=test(clf,X_train,y_train)
    acc_test,f1_test=test(clf,X_val,y_val)
    return acc_train,f1_train,acc_test,f1_test


parent_dir='/home/sonali/MLops/mnist/models_test'
dir='gamma_0.01'

def model_creation(X_train,y_train):
    gamma=0.01
    clf=svm.SVC(gamma=gamma)
    clf.fit(X_train,y_train)
    output = os.path.join(parent_dir,dir)
    try:
        os.mkdir(output)
    except:
        pass
        
    
    dump(clf, os.path.join(output,"model.joblib"))

    return output,clf





    







    




    
    #run_classification_experiment(data, expeted-model-file)

    #assert os.path.isfile(expected-model-file)

test_model_writing()
