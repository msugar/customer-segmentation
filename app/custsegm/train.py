# -*- coding: utf-8 -*-

"""Customer Personality Analysis
"""

import argparse
import logging
import os
from datetime import date
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class CustomerSegmentation:
    def __init__(self, model_path,
                 current_year=date.today().year,
                 age_cap=90, income_cap=600000,
                 debug=False):
        np.random.seed(42)
        self.model_path
        self.current_year = current_year
        self.age_cap = age_cap
        self.income_cap = income_cap
        self.training = training
        self.debug = debug
        logging.debug(f"Customer Segmentation" +
                      f" {model_path=}" +
                      f" {current_year=}" +
                      f" {age_cap=}" +
                      f" {income_cap=}" +
                      f" {debug=}.")


    def read_dataset_from_file(dataset_path):
        logging.debug(f"Read dataset from: {dataset_path}")
        data = pd.read_csv(dataset_path, sep="\t")
        return data


    def prepare(data, training=False):
        mode = "training" if training else "prediction"
        logging.debug(f"Prepare data for {mode}")

        # Remove missing values
        if training:
            data = data.dropna()

        # Create a feature out of "Dt_Customer" that indicates the number of
        # days a customer is registered in the firm's database
        data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"])
        max_date = data["Dt_Customer"].max()
        data["Customer_For"] = max_date
        data["Customer_For"] = (data["Customer_For"] - data["Dt_Customer"]).dt.days
        data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")

        # Age of customer as of today
        data["Age"] = self.current_year - data["Year_Birth"]

        # Total spendings on various items
        data["Spent"] = data[
            [
                "MntWines",
                "MntFruits",
                "MntMeatProducts",
                "MntFishProducts",
                "MntSweetProducts",
                "MntGoldProds",
            ]
        ].sum(axis=1)

        # Deriving living situation by marital status
        data["Living_With"] = data["Marital_Status"].replace(
            {
                "Married": "Partner",
                "Together": "Partner",
                "Absurd": "Alone",
                "Widow": "Alone",
                "YOLO": "Alone",
                "Divorced": "Alone",
                "Single": "Alone",
            }
        )

        # Feature indicating total children living in the household
        data["Children"] = data["Kidhome"] + data["Teenhome"]

        # Feature for total members in the householde
        data["Family_Size"] = (
            data["Living_With"].replace(
                {"Alone": 1, "Partner": 2}
            ) + data["Children"]
        )

        # Feature parenthood
        data["Is_Parent"] = np.where(data.Children > 0, 1, 0)

        # Segment education levels in three groups
        data["Education"] = data["Education"].replace(
            {
                "Basic": "Undergraduate",
                "2n Cycle": "Undergraduate",
                "Graduation": "Graduate",
                "Master": "Postgraduate",
                "PhD": "Postgraduate",
            }
        )

        # Drop some of the now redundant features
        redundant = ["Marital_Status", "Dt_Customer", "Year_Birth"]
        data = data.drop(redundant, axis=1)

        # Drop unused features
        unused = [
            "ID",
            "AcceptedCmp3",
            "AcceptedCmp4",
            "AcceptedCmp5",
            "AcceptedCmp1",
            "AcceptedCmp2",
            "Complain",
            "Response",
            "Z_CostContact",
            "Z_Revenue",
        ]
        data = data.drop(unused, axis=1, errors="ignore")

        if training:
            # Drop the outliers by setting a cap on age and income
            data = data[data["Age"] < self.age_cap]
            data = data[data["Income"] < self.income_cap]
            
        if self.debug:
            print("Prepared data:")
            print(data.info())
            print(data)

        return data


    def train(data, ncluster=4):
        logging.debug(f"Train {nclusters=}")
        
        data = prepare(data, training=True)

        #numerical = data.select_dtypes(include=["int64", "float64"]).columns
        categorical = data.select_dtypes(include=["object", "bool"]).columns
        logging.debug("Categorical variables in the dataset:",
                      categorical.values)

        transformer = ColumnTransformer(
            [
                ('oe', OrdinalEncoder(), categorical)
            ],
            remainder = "passthrough"
        )

        pipeline = Pipeline(steps=[
            ('tr', transformer),
            ('sts', StandardScaler()),
            ('km', KMeans(n_clusters=ncluster))
        ])

        pipeline.fit_transform(data)

        data["Clusters"] = pipeline["km"].labels_

        if debug:
            print("Trained data\n", data)
            print("Trained data stats\n", data.describe().T)

        joblib.dump(pipeline, model_path)


    def predict(data):
        logging.debug(f"Predict")
        data = prepare(data, training=False)
        pipeline = joblib.load("model.joblib")
        predictions = pipeline.predict(data)
        if debug:
            print("Predictions\n", predictions)
        return predictions


def train_and_evaluate(train_data_pattern, eval_data_pattern, test_data_pattern, export_dir, output_dir):
    data_path = "data/marketing_campaign.tsv"
    train_data, test_data = CustomerSegmentation.\
        read_dataset_from_file(dataset_path, random_state=42)
    cust_segm = CustomerSegmentation(random_state=42)
    cust_segm.train(train_data)
    cust_segm.predict(test_data.dropna())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bucket',
        help='Data will be read from gs://BUCKET/custsegm/data and output will be in gs://BUCKET/custsegm/train_output',
        required=True
    )
    parser.add_argument(
        '--develop',
        help='Train on a small subset in development',
        dest='develop',
        action='store_true'
    )
    parser.set_defaults(develop=False)
    
    # parse args
    args = parser.parse_args().__dict__
    logging.getLogger().setLevel(logging.INFO)
    
    # set top-level output directory for checkpoints, etc.
    BUCKET = args['bucket']
    OUTPUT_DIR = 'gs://{}/custsegm/train_output'.format(BUCKET)
    
    # The Vertex AI contract. If not running in Vertex AI Training, these will be None
    OUTPUT_MODEL_DIR = os.getenv("AIP_MODEL_DIR")  # or None
    TRAIN_DATA_PATTERN = os.getenv("AIP_TRAINING_DATA_URI")
    EVAL_DATA_PATTERN = os.getenv("AIP_VALIDATION_DATA_URI")
    TEST_DATA_PATTERN = os.getenv("AIP_TEST_DATA_URI")

    # During hyperparameter tuning, we need to make sure different trials don't clobber each other
    # https://cloud.google.com/ai-platform/training/docs/distributed-training-details#tf-config-format
    # This doesn't exist in Vertex AI
    # OUTPUT_DIR = os.path.join(
    #     OUTPUT_DIR,
    #     json.loads(
    #         os.environ.get('TF_CONFIG', '{}')
    #     ).get('task', {}).get('trial', '')
    # )
    if OUTPUT_MODEL_DIR:
        # convert gs://ai-analytics-solutions-dsongcp2/aiplatform-custom-job-2021-11-13-22:22:46.175/1/model/
        # to gs://ai-analytics-solutions-dsongcp2/aiplatform-custom-job-2021-11-13-22:22:46.175/1
        OUTPUT_DIR = os.path.join(
            os.path.dirname(OUTPUT_MODEL_DIR if OUTPUT_MODEL_DIR[-1] != '/' else OUTPUT_MODEL_DIR[:-1]),
            'train_output')
    logging.info('Writing checkpoints and other outputs to {}'.format(OUTPUT_DIR))
    
    # Set default values for the contract variables in case we are not running in Vertex AI Training
    if not OUTPUT_MODEL_DIR:
        OUTPUT_MODEL_DIR = os.path.join(OUTPUT_DIR,
                                        'export/flights_{}'.format(time.strftime("%Y%m%d-%H%M%S")))
    if not TRAIN_DATA_PATTERN:
        TRAIN_DATA_PATTERN = 'gs://{}/custsegm/data/train*'.format(BUCKET)
    if not EVAL_DATA_PATTERN:
        EVAL_DATA_PATTERN = 'gs://{}/custsegm/data/eval*'.format(BUCKET)
    logging.info('Exporting trained model to {}'.format(OUTPUT_MODEL_DIR))
    logging.info("Reading training data from {}".format(TRAIN_DATA_PATTERN))
    logging.info('Writing trained model to {}'.format(OUTPUT_MODEL_DIR))
    
    DEVELOP_MODE = args['develop']
    
    # run
    train_and_evaluate(TRAIN_DATA_PATTERN, EVAL_DATA_PATTERN, TEST_DATA_PATTERN, OUTPUT_MODEL_DIR, OUTPUT_DIR)

    logging.info("Done")
