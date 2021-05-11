import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
vect = joblib.load('netflix_vector.pkl')
clf = joblib.load('netflix_svm_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
     features =[x for x in request.form.values()]
     features = np.array(features)
     vector = vect.transform(features)
     predicted = clf.predict(vector)
     if predicted[0]==0:
      output='negative 😭'
     else:
      output='positive 😊'  
     return render_template('index.html',prediction_text='The review is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
