from typing import List, Tuple

import subprocess
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

import warnings


class Dataset():
    
    @staticmethod
    def read_train_test(uri: str,
                        test_size: float = 0.10,
                        random_state: int = 42
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
        
        # Read training dataset (assumed to fit in memory)
        # as a Tab-Separated-Values (*.tsv) file
        df = pd.read_csv(uri, sep="\t")
            
        # Remove missing values
        df = df.dropna()
        
        # Split dataset into train and test
        train_df, test_df = train_test_split(df,
                                             test_size=test_size,
                                             random_state=random_state)

        return train_df, test_df


    @staticmethod
    def read_train_test_from_default_gcs_bucket(test_size: float = 0.10,
                                                random_state: int = 42
                                               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        bashCommand = "gcloud config get-value project"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        if error:
            logging.error(f"error={error}")
            raise Exception("Could not find the default GCP project id. Is Google Cloud SDK installed?")
        project_id = output.decode('utf8').strip()
        logging.debug(f"project_id={project_id}")
        gcs_path_to_dataset = f"gs://{project_id}-bucket/custom-training/custsegm/data/marketing_campaign.csv"
        logging.debug(f"gcs_path_to_dataset={gcs_path_to_dataset}.")
        return Dataset.read_train_test(gcs_path_to_dataset, test_size, random_state)
