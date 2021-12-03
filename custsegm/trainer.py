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
from google.cloud.storage import Client as StorageClient, Blob
from google.cloud.logging import Client as LogClient
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class Trainer:
    def __init__(self,
                 project_id: str,
                 dataset_uri: str,
                 model_dir: str,
                 debug: bool = False):
        self.dataset_filename = "dataset.tsv"
        self.artifact_filename = "model.joblib"
        self.project_id = project_id
        self.dataset_uri = dataset_uri
        self.model_dir = model_dir
        self.debug = debug
        logging.debug(f"Trainer" +
                      f" dataset_uri={dataset_uri}" +
                      f" model_dir={model_dir}" +
                      f" debug={debug}.")


    def run(self) -> None: 
        logging.info("Custom training job started...")
        
        # Import the dataset
        if self.dataset_uri:
            if self.dataset_uri.startswith("gs://"):
                self.download_dataset_from_gcs()

        # Split dataset into training and test data
        logging.info(f"Loading dataset")
        train_data, test_data = CustomerSegmentation.\
            read_train_test_data(self.dataset_filename, test_size=0.1)  

        logging.info("Fitting model")
        trainee = CustomerSegmentation()
        pipeline = trainee.train(train_data)
        
        # Save model artifact to local filesystem (doesn't persist)
        logging.info(f"Saving fitted model")
        joblib.dump(pipeline, self.artifact_filename)
        
        # Export the model
        if self.model_dir:
            if self.model_dir.startswith("gs://"):
                self.upload_model_to_gcs()
                
        # Clean-up
        if self.dataset_uri:
            os.remove(self.dataset_filename)
        if self.model_dir:
            os.remove(self.artifact_filename)
        
        logging.info("Custom training job done")


    def download_dataset_from_gcs(self) -> None:
        """Downloads dataset from GCS
        """
        logging.info("Downloading dataset from GCS...")

        # Download dataset from Cloud Storage
        storage_path = self.dataset_uri
        client = StorageClient()
        blob = Blob.from_string(storage_path, client=client)
        blob.download_to_filename(self.dataset_filename)

        logging.info(f"Downloaded dataset from {storage_path}")


    def upload_model_to_gcs(self) -> None:
        """Uploads trained pipeline to GCS
        """
        logging.info("Uploading model artifact to GCS...")
        
        # Upload model artifact to Cloud Storage
        storage_path = os.path.join(self.model_dir, self.artifact_filename)
        client = StorageClient()
        blob = Blob.from_string(storage_path, client=client)
        blob.upload_from_filename(self.artifact_filename)
        
        logging.info(f"Uploaded model artifact to: {storage_path}")


# Define all the command line arguments your model can accept for training
if __name__ == "__main__":

    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cloud_logging_project_id",
        help="GCP project id for cloud logging.",
        type=str
    )
    args = parser.parse_args()

    # Explicit project selection:
    # See: https://cloud.google.com/vertex-ai/docs/training/code-requirements
    project_id = os.getenv("CLOUD_ML_PROJECT_ID")

    # Configure logging
    cloud_logging_project_id = args.cloud_logging_project_id or project_id
    if cloud_logging_project_id:
        # Set up the GCP logger
        client = LogClient(project=cloud_logging_project_id)
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

    dataset_uri = AIP_TRAINING_DATA_URI or "marketing_campaign.csv"
    model_dir = AIP_MODEL_DIR
    trainer = Trainer(project_id, dataset_uri, model_dir)
    trainer.run()