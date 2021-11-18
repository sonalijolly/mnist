from flask import Flask
from flask import request
#from mnist.utils import load
from joblib import dump, load
import numpy as np

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

best_model_path = "/home/sonali/MLops/mnist/models/s_1_tt_0.25_gamma_0.001/model.joblib"

@app.route("/predict", methods = ['POST'])
def predict():
    clf = load(best_model_path)
    input_json = request.json
    image = input_json['image']
    #print(image)
    #image = np.array(image)
    image = np.array(image).reshape(1,-1)
    predicted = clf.predict(image)
    return str(predicted[0])
