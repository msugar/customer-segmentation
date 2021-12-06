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
    @classmethod
    def setUpClass(cls):
        logging.getLogger().setLevel(logging.INFO)
        cls.model_path = "test_model.joblib"
        cls.train_data, cls.test_data = Dataset.read_train_test_from_default_gcs_bucket()
        
        cls.test_json_request = """
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


    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.model_path):
            os.remove(cls.model_path)


    def test_train_predict(self):
        sut = CustomerSegmentation(as_of_year=2021)
        sut.train(CustomerSegmentationTestCase.train_data)
        predictions = sut.predict(CustomerSegmentationTestCase.test_data)
        self.assertEqual(predictions.size, 222)


    def test_reusing_model(self):
        sut_1 = CustomerSegmentation(as_of_year=2021)
        pipeline = sut_1.train(CustomerSegmentationTestCase.train_data)
        
        sut_2 = CustomerSegmentation(pipeline, as_of_year=2021)
        predictions = sut_2.predict(CustomerSegmentationTestCase.test_data)
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
        sut.train(CustomerSegmentationTestCase.train_data)
        input = CustomerSegmentationTestCase.test_data.head(1)
        #print("Data Input:", input)
        predictions = sut.predict(input)
        #print("Data Predictions:", predictions)

        
    def test_prediction_from_cc_vertex_ai_json(self):
        a_json = json.loads(CustomerSegmentationTestCase.test_json_request)
        b_json = a_json['instances']
        input = pd.DataFrame(b_json)
        
        sut = CustomerSegmentation(as_of_year=2021)
        sut.train(CustomerSegmentationTestCase.train_data)
        
        #print("JSON Input:", input)
        predictions = sut.predict(input, preprocess=False)
        #print("JSON Predictions:", predictions.tolist())
        
        expected = [3, 3]
        self.assertListEqual(predictions.tolist(), expected)
        
        
    def test_predictor_predict_with_vertex_ai_payload(self):
        pred = Predictor.as_set_by_envvars()
        pred.ready()
        
        vertex_ai_input = json.loads(CustomerSegmentationTestCase.test_json_request)
        vertex_ai_output = pred.predict_from_vertex_ai(vertex_ai_input)
        predictions = vertex_ai_output["predictions"]
        
        expected = [3, 3]
        self.assertListEqual(predictions, expected)
        
        
    def test_predictor_predict_with_instances(self):
        pred = Predictor.as_set_by_envvars()
        pred.ready()
        
        vertex_ai_input = json.loads(CustomerSegmentationTestCase.test_json_request)["instances"]
        vertex_ai_output = pred.predict_from_vertex_ai(vertex_ai_input)
        predictions = vertex_ai_output["predictions"]
        
        expected = [3, 3]
        self.assertListEqual(predictions, expected)        

