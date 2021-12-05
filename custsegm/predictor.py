# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import joblib
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union
#import numpy.typing as npt

import pandas as pd
from google.cloud.storage import Client as StorageClient, Blob
from google.cloud.logging import Client as LogClient

class Predictor:
    def __init__(self,
                 model_dir: str,
                 artifact_filename: str):
        self.model_dir = model_dir
        # If you are in a live tutorial session, you might be using a shared
        # test account or project. To avoid name collisions between users on
        # resources created, you create a timestamp for each instance 
        # session, and append it onto the name of local resources you create
        #TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
        #self.artifact_filename = f"model-{TIMESTAMP}.joblib"
        if artifact_filename:
            self.artifact_filename = artifact_filename
        else:
            self.artifact_filename = "model-used-by-predictor.joblib"
        logging.debug(f"Predictor" +
                      f" model_dir={self.model_dir}" +
                      f" artifact_filename={self.artifact_filename}.")


    def ready(self):
        logging.debug("Readying predictor...")
        
        # Import the model
        if self.model_dir.startswith("gs://"):
            self.download_model_from_gcs()
            
        # Load model artifact from local filesystem
        logging.debug(f"Loading model from local file {self.artifact_filename}")
        self.pipeline = joblib.load(self.artifact_filename)
        
        logging.debug("Predictor is ready.")


    def download_model_from_gcs(self) -> None:
        """Downloads model.joblib from GCS
        """
        # Download model artifact from Cloud Storage
        storage_path = os.path.join(self.model_dir, "model.joblib") # file name required by Vertex AI
        logging.debug(f"Downloading model artifact from GCS bucket {storage_path} to local file {self.artifact_filename}")
        client = StorageClient()
        blob = Blob.from_string(storage_path, client=client)
        blob.download_to_filename(self.artifact_filename)
        
        
    def predict_from_dataframe(self, df: pd.DataFrame):
        logging.debug(f"Predictor df << {df}")
        predictions = self.pipeline.predict(df)
        logging.debug(f"Predictor df >> {predictions}")
        return predictions


    def predict_from_vertex_ai(self, vertex_ai_input: Union[List, Dict[str, List]]) -> Dict:
        logging.debug(f"Predictor vertex ai << {vertex_ai_input}")
        if isinstance(vertex_ai_input, dict) and vertex_ai_input["instances"]:
            instances = vertex_ai_input["instances"]
        else:
            instances = vertex_ai_input
        df = pd.DataFrame(instances)
        predictions = self.predict_from_dataframe(df).tolist()
        vertex_ai_output = { "predictions": predictions }
        logging.debug(f"Predictor vertex ai >> {vertex_ai_output}")
        return vertex_ai_output
    
    
    @staticmethod
    def as_set_by_envvars():
        AIP_MODEL_DIR = os.getenv('AIP_MODEL_DIR')
        if not AIP_MODEL_DIR:
            raise ValueError("AIP_MODEL_DIR not set.")
        logging.debug(f"AIP_MODEL_DIR={AIP_MODEL_DIR}")
        
        predictor = Predictor(AIP_MODEL_DIR)
        return predictor