from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
import numpy as np

digits = datasets.load_digits()
X,Y = digits.images,digits.target 



def create_split(data,y):
        X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=0.15, shuffle=False)
        
        X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.50, shuffle=False)
        
        return X_train, X_test, y_train, y_test, X_val, y_val
X_train, X_test, y_train, y_test, X_val, y_val=create_split(X,Y)

gamma_parameter = [0.01,0.1,1]
max_iter=[100,500,1000,1]
kernel=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']



train_firstrun=[]
dev_firstrun=[]
test_firstrun=[]



for gamma_p in gamma_parameter:
    for itert in max_iter:
        for k in kernel:
            #model creation
                clf = svm.SVC(gamma=gamma_p, max_iter=itert, kernel=k)
                
                X_train = X_train.reshape((len(X_train), -1))
                X_val =  X_val.reshape((len(X_val), -1))
                X_test =  X_val.reshape((len(X_test), -1))
                #print(y_train.shape)

                clf.fit(X_train,y_train) 
                acc_val=clf.score(X_val, y_val)
                acc_train=clf.score(X_train, y_train)
                acc_test=clf.score(X_test, y_test)

                test_firstrun.append(acc_test)
                train_firstrun.append(acc_train)
                dev_firstrun.append(acc_val)
print(train_firstrun)               





