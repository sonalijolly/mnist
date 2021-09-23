from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
import numpy as np

digits = datasets.load_digits()

def create_split(data, sp):
        X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=sp, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(
        data, digits.target, test_size=sp, shuffle=False)
        #print(np.array(X_train).shape)
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

def report(model,y_test,preds):
    print(f"Classification report for classifier {model}:\n"
      f"{metrics.classification_report(y_test, preds)}\n")