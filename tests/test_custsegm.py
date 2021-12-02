# tests/test_custsegm.py

import unittest
import logging
import os
from custsegm.custsegm import CustomerSegmentation

class CustomerSegmentationTestCase(unittest.TestCase):
    def setUp(self):
        logging.getLogger().setLevel(logging.INFO)
        self.model_path = "tests/custsegm_model.joblib"
        dataset_uri = "data/marketing_campaign.tsv"
        self.train_data, self.test_data = CustomerSegmentation.\
            read_train_test_data(dataset_uri, test_size=0.10)


    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)


    def test_train_predict(self):
        sut = CustomerSegmentation(as_of_year=2021)
        sut.train(self.train_data)
        predictions = sut.predict(self.test_data)
        self.assertEqual(predictions.size, 222)


    def test_save_and_load_model(self):
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
