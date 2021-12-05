# -*- coding: utf-8 -*-

"""Customer Personality Analysis
"""

import logging
import os
from typing import List, Tuple
from datetime import date
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator

#import sys
import warnings

#if not sys.warnoptions:
#    warnings.simplefilter("ignore")

from pandas.core.common import SettingWithCopyWarning


class CustomerSegmentation:
    def __init__(self, 
                 pipeline: Pipeline = None,
                 as_of_year: int = None,
                 age_cap: int = 90,
                 income_cap: int = 600000,
                 debug: bool = False):
        self.pipeline = pipeline
        self.as_of_year = as_of_year or date.today().year
        self.age_cap = age_cap
        self.income_cap = income_cap
        self.debug = debug
            
        logging.debug(f"Customer Segmentation" +
                      f" as_of_year={as_of_year}" +
                      f" age_cap={age_cap}" +
                      f" income_cap={income_cap}" +
                      f" debug={debug}.")


    def preprocess(self, data: pd.DataFrame, training: bool = False):
        mode = "training" if training else "prediction"
        logging.debug(f"Preprocess data for {mode}")

        if training:
            data = data.dropna()

        # Create a feature out of "Dt_Customer" that indicates the number of
        # days a customer is registered in the firm's database
        warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
        data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"])
        max_date = data["Dt_Customer"].max()
        data["Customer_For"] = max_date
        data["Customer_For"] = (data["Customer_For"] - data["Dt_Customer"]).dt.days
        data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")

        # Age of customer as of today
        data["Age"] = self.as_of_year - data["Year_Birth"]

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
            
        #if self.debug:
        #    print(f"Preprocessed {mode} data:")
        #    print(data.info())
        #    print(data)

        return data


    def train(self, data: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> Pipeline:
        logging.debug(f"Train n_clusters={n_clusters}")
        
        data = self.preprocess(data, training=True)

        #numerical = data.select_dtypes(include=["int64", "float64"]).columns
        categorical = data.select_dtypes(include=["object", "bool"]).columns
        logging.debug(f"Categorical variables in the dataset: " +
                      f"{categorical.values}")

        transformer = ColumnTransformer(
            [
                ('oe', OrdinalEncoder(), categorical)
            ],
            remainder = "passthrough"
        )

        pipeline = Pipeline(steps=[
            ('tr', transformer),
            ('sts', StandardScaler()),
            ('km', KMeans(n_clusters=n_clusters, random_state=random_state))
        ])

        pipeline.fit_transform(data)

        data["Clusters"] = pipeline["km"].labels_

        #if self.debug:
        #    print("Trained data\n", data)
        #    print("Trained data stats\n", data.describe().T)

        self.pipeline = pipeline
        return pipeline


    def predict(self, data: pd.DataFrame, preprocess=True):
        logging.debug(f"Predict")
        
        if preprocess:
            data = self.preprocess(data, training=False)
        
        if self.debug:
            print("Prediction input:")
            #json_string = data.head(1).reset_index().to_json(orient='records')
            #print(json_string)
            print(data)
        
        predictions = self.pipeline.predict(data)
        
        if self.debug:
            print("Prediction output:")
            print(predictions)
            
        return predictions
