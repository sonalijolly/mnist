

print(__doc__)



# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)





digits = datasets.load_digits()
print(digits.images.shape)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)


def testf(digits, shape, split):
    n_samples = len(digits.images)
    X,Y = digits.images,digits.target 

    sample1 = np.resize(digits.images, (n_samples,shape))
    

    clf = svm.SVC(gamma=0.00001)


    X_train, X_test, y_train, y_test = train_test_split(
        sample1, digits.target, test_size=split, shuffle=False)

    
    
    
    clf.fit(X_train, y_train.reshape(-1,1))

    
    predicted = clf.predict(X_test)

    
    accuracy = metrics.accuracy_score(y_test, predicted)
    
    return accuracy

shape = 256
split = 0.75
a= testf(digits,shape , split)
print("16*16   ","75:25   ",a)

shape = 256
split = 0.5
a= testf(digits,shape , split)
print("16*16   ","50:50   ", a)

shape = 256
split = 0.25
a= testf(digits,shape , split)
print("16*16   ","25:75   ", a)


shape = 1024    
split = 0.75
a= testf(digits,shape , split)
print("32*32  ","75:25   ",a)

shape = 1024    
split = 0.5
a= testf(digits,shape , split)
print("32*32   ","50:50   ", a)

shape = 1024
split = 0.25
a= testf(digits,shape , split)
print("32*32    ","25:75   ", a)

shape = 4096    
split = 0.75
a= testf(digits,shape , split)
print("64*64   ","75:25   ", a)

shape = 4096    
split = 0.5
a= testf(digits,shape , split)
print("64*64   ","75:25   ", a)

shape = 4096
split = 0.25
a= testf(digits,shape , split)
print("64*64   ","75:25   ", a)

