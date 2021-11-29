# tests/test_custsegm.py

import unittest
import logging
import os
from custsegm.custsegm import CustomerSegmentation

class CustomerSegmentationTestCase(unittest.TestCase):
    def setUp(self):
        logging.getLogger().setLevel(logging.INFO)
        self.model_path = "tests/custsegm_model.joblib"
        self.dataset_path = "data/marketing_campaign.tsv"


    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)


    def test_train_predict(self):
        train_data, test_data = CustomerSegmentation.\
            read_dataset_from_file(self.dataset_path, random_state=42)

        sut = CustomerSegmentation(random_state=42, debug=False)
        sut.train(train_data)
        predictions = sut.predict(test_data)
        self.assertEqual(predictions.size, 222)


    def test_save_and_load_model(self):
        train_data, test_data = CustomerSegmentation.\
            read_dataset_from_file(self.dataset_path, random_state=42)

        sut_1 = CustomerSegmentation(random_state=42, debug=False)
        sut_1.train(train_data)
        sut_1.save_model(self.model_path)
        
        sut_2 = CustomerSegmentation(random_state=42)
        sut_2.load_model(self.model_path)
        predictions = sut_2.predict(test_data)
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
