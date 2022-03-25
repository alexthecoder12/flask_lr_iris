import joblib
import numpy as np
from flask import Flask, request, render_template, jsonify

model = None
app = Flask(__name__)

def load_model():
    '''
    Load the model once and reuse in every call.
    '''
    global model
    with open('model.sav', 'rb') as f:
        model = joblib.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [np.array([float(x) for x in request.form.values()])]
    prediction = model.predict(features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='IRIS class should be : {}'.format(output))

@app.route('/results', methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = str(prediction[0])

    return jsonify(output)

if __name__ == '__main__':
    load_model()  ## load model at the beginning once only ##
    app.run(host='0.0.0.0', port=5000) ## change to port 80 when running on AWS EC2 ##