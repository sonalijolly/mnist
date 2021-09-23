import matplotlib.pyplot as plt
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import DataConversionWarning
from skimage.transform import rescale, resize, downscale_local_mean
from utils import create_split,rescale_resize,classfreport

import sys
import os
import shutil
from joblib import dump, load
import numpy as np
import math

candidate_model =[]

digits = datasets.load_digits()

def model_score(model, X_val, y_val, gamma, sp, sh):
    acc=model.score(X_val, y_val)
    if acc < 0.11:
        pass
        print("Skip shape {} split {} gamma {}". format(sh,sp,gamma))
    candidate = {
        'acc' :   acc,
        'gamma' : gamma,
        'split' : sp,
        'shape' : sh
        }
    candidate_model.append(candidate)

def test(X,best_model_folder, shape,split):
    data = data_preprocess(X, shape)
    X_train, X_test, y_train, y_test, X_val, y_val = create_split(data, split)

    clf = load(os.path.join(best_model_folder,"model.joblib"))
    predicted = clf.predict(X_test)
    classfreport(clf, y_test, predicted) # classification report function   

mydir = "/home/sonali/MLops/mnist/models"
if os.path.exists(mydir):
    shutil.rmtree(mydir)
X,Y = digits.images,digits.target 
split_parameter = [0.25, 0.5, 0.75]
shape_parameter = [0.5, 1, 2, 4]
gamma_parameter = [1e-7,1e-5,1e-3,0.01,0.1,1]

for sh in shape_parameter:
    for sp in split_parameter:
        for gamma_p in gamma_parameter:
            #data preprocessing for rescale and reshape
            data = rescale_resize(X,sh)
            #model creation
            clf = svm.SVC(gamma=gamma_p)
            # data split function
            X_train, X_test, y_train, y_test, X_val, y_val = create_split(data, sp) 
            #model training
            #print(y_train.shape)
            #y_train = y_train.reshape(-1,1)
            clf.fit(X_train,y_train) 
            #appending model candidates
            model_score(clf, X_val, y_val, gamma_p, sp, sh)
            output = "/home/sonali/MLops/mnist/models/"+"s_{}_tt_{}_gamma_{}".format(sh,sp,gamma_p)
            #saving models 
            os.makedirs(output)
            dump(clf, os.path.join(output,"model.joblib"))
#finding best model on accuracy
best_model = max(candidate_model, key = lambda x: x['acc'])  

#folder for best model
bestmodel_shape=best_model["shape"]
bestmodel_split=best_model["split"]
bestmodel_gamma=best_model["gamma"]

best_model_folder = "/home/sonali/MLops/mnist/models/s_{}_tt_{}_gamma_{}".format(bestmodel_shape,bestmodel_split,bestmodel_gamma)


test(X,best_model_folder, bestmodel_shape,bestmodel_split)
