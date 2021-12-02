# -*- coding: utf-8 -*-

"""Customer Personality Analysis
"""

import argparse
import logging
import os
import joblib
from typing import List, Tuple

from custsegm.custsegm import CustomerSegmentation

import pandas as pd
from google.cloud import storage
from google.cloud.logging import Client as LogClient
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class Trainer:
    def __init__(self,
                 project_id: str,
                 dataset_uri: str,
                 model_uri: str,
                 debug: bool = False):
        
        self.artifact_filename = 'model/custsegm_model.joblib'
        self.project_id = project_id
        self.dataset_uri = dataset_uri
        self.model_uri = model_uri
        self.debug = debug
        logging.debug(f"Trainer" +
                      f" dataset_uri={dataset_uri}" +
                      f" model_uri={model_uri}" +
                      f" debug={debug}.")


    def run(self) -> None: 
        logging.info("Custom training job started...")
        
        logging.info(f"Reading dataset from {self.dataset_uri}")
        train_data, test_data = CustomerSegmentation.\
            read_train_test_data(self.dataset_uri, test_size=0.1)          

        logging.info(f"Fitting model")
        trainee = CustomerSegmentation()
        pipeline = trainee.train(train_data)
        
        # Save model artifact to local filesystem (doesn't persist)
        logging.info(f"Saving fitted model to local file {self.artifact_filename}")
        joblib.dump(pipeline, self.artifact_filename)
        
        if self.model_uri:
            if self.model_uri.startswith("gs://"):
                self.export_model_to_gcs()
        
        logging.info("Done")


    def export_model_to_gcs(self) -> None:
        """Exports trained pipeline to GCS
        """
        logging.info("Exporting model artifact to GCS...")
        
        # Upload model artifact to Cloud Storage
        storage_path = os.path.join(self.model_uri, self.artifact_filename)
        client = storage.Client()
        blob = storage.blob.Blob.from_string(storage_path, client=client)
        blob.upload_from_filename(self.artifact_filename)
        
        logging.info(f"Exported model artifact to: {storage_path}")


# Define all the command line arguments your model can accept for training
if __name__ == "__main__":

    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_id",
        help="GCP project id for cloud logging.",
        type=str
    )
    args = parser.parse_args()

    # Explicit project selection:
    # See: https://cloud.google.com/vertex-ai/docs/training/code-requirements
    project_id = args.project_id
    if project_id is None:
        project_id = os.getenv("CLOUD_ML_PROJECT_ID")

    if project_id:
        # Set up the GCP logger
        client = LogClient(project=project_id)
        client.setup_logging(log_level=logging.INFO)
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Vertex AI custom training enables you to train on Vertex AI datasets 
    # and produce Vertex AI models. To do so your script must adhere to the
    # following contract:

    # (If you use managed datasets, which is optional) 
    # It must read datasets from the environment variables populated by the
    # training service:
    # See: https://cloud.google.com/vertex-ai/docs/training/using-managed-datasets
    AIP_DATA_FORMAT = os.getenv('AIP_DATA_FORMAT') # provides format of data: csv, jsonl, or bigquery
    AIP_TRAINING_DATA_URI = os.getenv('AIP_TRAINING_DATA_URI') # uri to training split
    AIP_VALIDATION_DATA_URI = os.getenv('AIP_VALIDATION_DATA_URI') # uri to validation split
    AIP_TEST_DATA_URI = os.getenv('AIP_TEST_DATA_URI') # uri to test split
        
    # It must write the model artifact to the environment variable populated
    # by the traing service:
    AIP_MODEL_DIR = os.getenv('AIP_MODEL_DIR')

    training_data_uri = AIP_TRAINING_DATA_URI or "data/marketing_campaign.tsv"
    model_uri = AIP_MODEL_DIR
        
    trainer = Trainer(project_id, training_data_uri, model_uri)
    trainer.run()