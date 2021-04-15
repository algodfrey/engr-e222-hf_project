from sklearn import datasets
from joblib import load
import numpy as np
import json

#load the model
hf_model = load('hf_model.pkl') 


def prediction(parameters):
    dummy = np.array(parameters)
    dummyT = dummy.reshape(1,-1)
    prediction = hf_model.predict(dummyT) # makes a prediction based on given input for parameters, which was stored in dummyT
    name_str = str(prediction[0])

    return name_str