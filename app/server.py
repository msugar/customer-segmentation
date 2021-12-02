import pandas as pd
from flask import Flask, jsonify, request
#import tensorflow as tf
from custsegm.custsegm import CustomerSegmentation


app = Flask(__name__)

classifier = CustomerSegmentation(random_state=42)
classifier.load_model("model/custsegm_model.joblib")

@app.route('/predict',methods=['POST','GET'])
def predict():
    req = request.json.get('instances')
    
    education = req[0]['Education']
    income = req[0]['Income']
    kidhome = req[0]['Kidhome']
    teenhome = req[0]['Teenhome']
    recency = req[0]['Recency']
    mnt_wines = req[0]['MntWines']
    mnt_fruits = req[0]['MntFruits']
    mnt_meat_products = req[0]['MntMeatProducts']
    mnt_fish_products = req[0]['MntFishProducts']
    mnt_sweet_products = req[0]['MntSweetProducts']
    mnt_gold_prods = req[0]['MntGoldProds']
    num_deals_purchases = req[0]['NumDealsPurchases']
    num_web_purchases = req[0]['NumWebPurchases']
    num_catalog_purchases = req[0]['NumCatalogPurchases']
    num_store_purchases = req[0]['NumStorePurchases']
    num_web_visits_month = req[0]['NumWebVisitsMonth']
    customer_for = req[0]['Customer_For']
    age = req[0]['Age']
    spent = req[0]['Spent']
    living_with = req[0]['Living_With']
    children = req[0]['Children']
    family_size = req[0]['Family_Size']
    is_parent = req[0]['Is_Parent']

    #predict
    prediction = model.predict(vector)

    #postprocessing
    value = post_process.postprocess(list(prediction[0])) 
    output = {'predictions':[
        {
           'label' : value
        }
        ]
        }
    return jsonify(output)

@app.route('/healthz')
def healthz():
    return "OK"


if __name__=='__main__':
    app.run(host='0.0.0.0')


