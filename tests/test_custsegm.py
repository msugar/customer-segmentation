# tests/test_custsegm.py

import unittest
import logging
import os
import json
import pandas as pd
from custsegm.custsegm import CustomerSegmentation
from custsegm.dataset import Dataset
from custsegm.predictor import Predictor


class CustomerSegmentationTestCase(unittest.TestCase):
    def setUp(self):
        logging.getLogger().setLevel(logging.INFO)
        self.model_path = "test_model.joblib"
        dataset_uri = "gs://term-project-331703-bucket/custom-training/custsegm/data/marketing_campaign.csv"
        self.train_data, self.test_data = Dataset.read_train_test(dataset_uri)
        
        self.json_str = """
            {
              "instances": [
                  {
                    "Education": "Postgraduate",
                    "Income": 52597,
                    "Kidhome": 0,
                    "Teenhome": 1,
                    "Recency": 69,
                    "MntWines": 492,
                    "MntFruits": 0,
                    "MntMeatProducts": 37,
                    "MntFishProducts": 7,
                    "MntSweetProducts": 0,
                    "MntGoldProds": 42,
                    "NumDealsPurchases": 3,
                    "NumWebPurchases": 6,
                    "NumCatalogPurchases": 3,
                    "NumStorePurchases": 8,
                    "NumWebVisitsMonth": 5,
                    "Customer_For": 153,
                    "Age": 59,
                    "Spent": 578,
                    "Living_With": "Alone",
                    "Children": 1,
                    "Family_Size": 2,
                    "Is_Parent": 1
                  },
                  {
                    "Education": "Graduate",
                    "Income": 75433,
                    "Kidhome": 1,
                    "Teenhome": 0,
                    "Recency": 28,
                    "MntWines": 800,
                    "MntFruits": 0,
                    "MntMeatProducts": 297,
                    "MntFishProducts": 0,
                    "MntSweetProducts": 34,
                    "MntGoldProds": 57,
                    "NumDealsPurchases": 2,
                    "NumWebPurchases": 2,
                    "NumCatalogPurchases": 5,
                    "NumStorePurchases": 10,
                    "NumWebVisitsMonth": 6,
                    "Customer_For": 215,
                    "Age": 32,
                    "Spent": 1188,
                    "Living_With": "Partner",
                    "Children": 1,
                    "Family_Size": 3,
                    "Is_Parent": 1
                  }
              ],
              "parameters": []
            }
            """
        
        super().setUp()


    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)


    def test_train_predict(self):
        sut = CustomerSegmentation(as_of_year=2021)
        sut.train(self.train_data)
        predictions = sut.predict(self.test_data)
        self.assertEqual(predictions.size, 222)


    def test_reusing_model(self):
        sut_1 = CustomerSegmentation(as_of_year=2021)
        pipeline = sut_1.train(self.train_data)
        
        sut_2 = CustomerSegmentation(pipeline, as_of_year=2021)
        predictions = sut_2.predict(self.test_data)
        expected = [3, 0, 2, 3, 0, 1, 3, 3, 2, 1, 2, 0, 2, 3, 2, 1, 0, 1, 2,
                    1, 3, 3, 2, 0, 3, 3, 1, 0, 1, 1, 0, 1, 0, 0, 0, 2, 1, 2,
                    1, 1, 1, 0, 1, 3, 3, 2, 3, 0, 0, 0, 3, 2, 0, 3, 3, 1, 1,
                    3, 2, 0, 1, 0, 0, 0, 1, 3, 0, 1, 1, 2, 1, 3, 2, 2, 2, 2,
                    3, 1, 0, 3, 3, 1, 2, 1, 3, 0, 3, 0, 3, 2, 3, 0, 1, 0, 1,
                    3, 1, 3, 2, 0, 3, 2, 0, 3, 3, 0, 3, 0, 1, 0, 2, 3, 1, 1,
                    3, 0, 2, 2, 1, 1, 3, 3, 3, 3, 1, 0, 2, 3, 2, 0, 0, 2, 1,
                    2, 2, 3, 0, 0, 1, 1, 0, 2, 0, 0, 1, 2, 3, 1, 2, 3, 2, 0,
                    0, 3, 1, 2, 2, 0, 0, 0, 0, 0, 0, 3, 1, 1, 2, 3, 1, 3, 3,
                    1, 2, 0, 3, 3, 0, 3, 2, 1, 0, 0, 2, 2, 0, 0, 1, 0, 2, 0,
                    3, 3, 2, 0, 2, 0, 2, 0, 3, 1, 3, 2, 2, 1, 0, 3, 1, 1, 0,
                    2, 0, 0, 3, 0, 0, 0, 2, 2, 2, 3, 1, 2]
        self.assertListEqual(predictions.tolist(), expected)

        
    def test_prediction_input(self):
        sut = CustomerSegmentation(as_of_year=2021)
        sut.train(self.train_data)
        input = self.test_data.head(1)
        #print("Data Input:", input)
        predictions = sut.predict(input)
        #print("Data Predictions:", predictions)

        
    def test_prediction_from_cc_vertex_ai_json(self):
        a_json = json.loads(self.json_str)
        b_json = a_json['instances']
        input = pd.DataFrame(b_json)
        
        sut = CustomerSegmentation(as_of_year=2021)
        sut.train(self.train_data)
        
        #print("JSON Input:", input)
        predictions = sut.predict(input, preprocess=False)
        #print("JSON Predictions:", predictions.tolist())
        
        expected = [3, 3]
        self.assertListEqual(predictions.tolist(), expected)
        
        
    def test_predictor_predict_for_vertex_ai(self):
        pred = Predictor("term-project-331703", "gs://term-project-331703-bucket/custom-training/custsegm/model")
        pred.ready()
        
        vertex_ai_input = json.loads(self.json_str)
        #input = vertex_ai_input["instances"]
        #df0 = pd.DataFrame(input)
        #print("df0", df0)

        #predictions = pred.predict_from_dataframe(df0)
        vertex_ai_output = pred.predict_from_vertex_ai(vertex_ai_input)
        predictions = vertex_ai_output["predictions"]
        
        expected = [3, 3]
        self.assertListEqual(predictions, expected)
        
        
    def test_predictor_predict_for_vertex_ai2(self):
        pred = Predictor("term-project-331703", "gs://term-project-331703-bucket/custom-training/custsegm/model")
        pred.ready()
        
        vertex_ai_input = json.loads(self.json_str)["instances"]
        #input = vertex_ai_input["instances"]
        #df0 = pd.DataFrame(input)
        #print("df0", df0)

        #predictions = pred.predict_from_dataframe(df0)
        vertex_ai_output = pred.predict_from_vertex_ai(vertex_ai_input)
        predictions = vertex_ai_output["predictions"]
        
        expected = [3, 3]
        self.assertListEqual(predictions, expected)        

