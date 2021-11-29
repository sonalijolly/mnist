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
gamma_p = [1e-7,1e-5,1e-3,0.01,0.1,1]
depth = [5,10,50,100,500]
split = [0.15, 0.25, 0.40, 0.75, 0.1]
#list
acc_svm=[]
bestparam_svm=[]
acc_tree=[]
bestparam_tree=[]


def create_split(data, sp):
        X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=sp, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(
        data, digits.target, test_size=sp, shuffle=False)
        #print(np.array(X_train).shape)
        return X_train, X_test, y_train, y_test, X_val, y_val



def acc_cal_svm(gamma_p, sp):
  clf = svm.SVC(gamma=gamma_p)
  X_train, X_test, y_train, y_test, X_val, y_val = create_split(data, sp)
  clf.fit(X_train,y_train)
  acc = clf.score(X_val,y_val)
  return acc

def acc_cal_tree(depth, sp):
  treeclassifier = tree.DecisionTreeClassifier(max_depth = depth)
  X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=sp, shuffle=False)
    
  X_train, X_val, y_train, y_val = train_test_split(
        data, digits.target, test_size=sp, shuffle=True)
  
  treeclassifier.fit(X_train,y_train)
  predicted = treeclassifier.predict(X_val)
  acc = metrics.accuracy_score(y_val, predicted)
  #print(acc)
  return acc


for i in split:
  #making temporary list for storing acc
  temp_acc=[]
  for j in gamma_p:
    acc=acc_cal_svm(j,i)
    temp_acc.append(acc)

  max_acc_index=temp_acc.index(max(temp_acc))
  max_acc=max(temp_acc)
  best_gamma=gamma_p[max_acc_index]
  bestparam_svm.append(best_gamma)
  acc_svm.append(max_acc)


for i in split:
  #making temporary list for storing acc
  temp_acc=[]
  for j in depth:
    acc=acc_cal_tree(j,i)
    temp_acc.append(acc)

  max_acc_index=temp_acc.index(max(temp_acc))
  max_acc=max(temp_acc)
  best_depth=depth[max_acc_index]

  bestparam_tree.append(best_depth)
  acc_tree.append(max_acc)


  

  




#creation of dataframe 

data = {"split":split, "optimal_gamma": bestparam_svm, "acc_svm" : acc_svm, "optimal_depth": bestparam_tree, "acc_decisiontree": acc_tree}
df = pd.DataFrame(data)

mean_svm=str(df["acc_svm"].mean())
mean_tree=str(df["acc_decisiontree"].mean())


std_svm = str(round(df["acc_svm"].std(),4))

std_tree = str(round(df["acc_decisiontree"].std(),4))

plusminus_symbol="\u00B1"



              
ms = mean_svm+plusminus_symbol+std_svm
md = mean_tree+plusminus_symbol+std_tree
s = pd.Series([' ',' ', ms, ' ', md], index = ['split', 'optimal_gamma','acc_svm', 'optimal_depth', 'acc_decisiontree'])
df = df.append(s, ignore_index = True)
print(df)





