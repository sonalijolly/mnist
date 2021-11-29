from flask import Flask
from flask import request
#from mnist.utils import load
from joblib import dump, load
import numpy as np

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

best_model_path = "/home/sonali/mnist/models_svm/gamma_0.01/model.joblib"
best_dt_path="/home/sonali/mnist/models_dt/depth_100/model.joblib"


@app.route("/predict_svm", methods = ['POST'])
def predict():
    clf = load(best_model_path)
    input_json = request.json
    image = input_json['image']
    #print(image)
    #image = np.array(image)
    image = np.array(image).reshape(1,-1)
    predicted = clf.predict(image)
    return str(predicted[0])

# for decision tree

@app.route("/predict_dt", methods = ['POST'])
def predict_dt():
    clf = load(best_dt_path)
    input_json = request.json
    image = input_json['image']
    #print(image)
    #image = np.array(image)
    image = np.array(image).reshape(1,-1)
    predicted = clf.predict(image)
    return str(predicted[0])


def main():
    app.run()
if __name__=="__main__":
    main()

