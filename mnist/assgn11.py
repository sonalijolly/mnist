import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn import datasets, svm, metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sys
import os
import shutil
from joblib import dump, load
import numpy as np
import math
#from utils import create_splits, preprocess, report, classification_model

digits = datasets.load_digits()
X,Y = digits.images, digits.target

f1_macro_list = []

mydir="/home/sonali/mnist/svm_models_train"

def create_split(data,y):
        X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=0.20, shuffle=False)
        
        X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.50, shuffle=False)
        
        return X_train, X_test, y_train, y_test, X_val, y_val
X_train, X_test, y_train, y_test, X_val, y_val=create_split(X,Y)

if os.path.exists(mydir):
    shutil.rmtree(mydir)



split_parameter = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]
# for creating sub -folder 

for sp in split_parameter:
    os.makedirs(os.path.join(mydir, str(sp)))


parames = [1e-7,1e-5,1e-3,0.01,0.1,1]

def model_score(model, X_val, y_val, gamma, sp):
    acc=model.score(X_val, y_val)

    # for model prediction
    pred=model.predict(X_val)

    f1 = metrics.f1_score(y_val, pred, average = 'macro' )

    candidate = {
        'acc' :   acc,
        'gamma' : gamma,
        'split' : sp,
        'f1':f1        
        }
    #candidate_model.append(candidate)
    return candidate

def run_classification_experiment(X_train,y_train,X_val,y_val,gamma_p, sp,  mydir):
    clf=svm.SVC(gamma=gamma_p,max_iter=1000)
    clf.fit(X_train,y_train)
    model_candidate = model_score(clf,X_val, y_val,gamma_p, sp) # validation function
    #print(model_candidate)
    output = mydir+"split_{}_gamma{}".format(sp, gamma_p)
    os.makedirs(output)
    dump(clf, os.path.join(output,"model.joblib"))
    return model_candidate


for sp in split_parameter:
      candidate =[]
      for gamma_p in parames:
            split_size=int(sp*len(X_train))
            X_train_new,Y_train_new = X_train[:split_size],y_train[:split_size]
            X_train_new,X_val=X_train_new.reshape((len(X_train_new), -1)),X_val.reshape((len(X_val), -1))
            mydir = mydir + "/{}/".format(sp)
            candidate.append(run_classification_experiment(X_train_new, Y_train_new, X_val, y_val, gamma_p, sp,  mydir))

      best_model = max(candidate, key = lambda x: x['f1'])  
      #folder for best model
      #folder for best model
      bestmodel_acc=best_model["acc"]
      bestmodel_split=best_model["split"]
      bestmodel_gamma=best_model["gamma"]
      bestmodel_f1=best_model["f1"]
      f1_macro_list.append(bestmodel_f1)
      print(best_model)

plt.plot(split_parameter,f1_macro_list)
plt.title("training size vs F1")
plt.xlabel("training size")
plt.ylabel("F1")
plt.show()







