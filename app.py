import pickle
from flask import Flask,request,app,jsonify,render_template
import numpy as np
import pandas as pd


app=Flask(__name__)
# Load the model
model=pickle.load(open('vehicle_pred_model1.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json['data']  
    for key, value in data.items():
        if isinstance(value, np.float32):
            data[key] = float(value)
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=np.array(list(data.values())).reshape(1,-1)
    output=model.predict(new_data)
    print(output[0])
    return jsonify(float(output[0]))

if __name__=="__main__":
    app.run(debug=True)
