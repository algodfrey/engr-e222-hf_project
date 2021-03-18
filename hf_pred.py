from sklearn import datasets
from joblib import load
import numpy as np
import json

#load the model

hf_model = load('hf_model.pkl') 

def prediction(age, anemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time):
    dummy = np.array([age, anemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time])
    dummyT = dummy.reshape(1,-1)
    r = dummy.shape
    t = dummyT.shape
    r_str = json.dumps(r)
    t_str = json.dumps(t)
    prediction = hf_model.predict(dummyT)
    name = class_names[prediction]
    name = name.tolist()
    name_str = json.dumps(name)
    str = [t_str, r_str, name_str]
    return str