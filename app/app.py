from flask import Flask, jsonify, request

import os
import json
import pandas as pd

from custsegm.predictor import Predictor


app = Flask(__name__)

predictor = Predictor.as_set_by_envvars()
predictor.ready()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return predict()
    else:
        return healthz()
    
@app.route('/predict',methods=['POST','GET'])
def predict():
    try:
        req = request.json.get('instances')

        #predict
        vertex_ai_input = req
        vertex_ai_output = predictor.predict_from_vertex_ai(vertex_ai_input)

        return jsonify(vertex_ai_output)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/healthz')
def healthz():
    return jsonify("OK")


if __name__=='__main__':
    #app.run(host='0.0.0.0')
    app.run(debug=True)
