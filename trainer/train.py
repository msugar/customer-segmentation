# -*- coding: utf-8 -*-

"""Customer Personality Analysis

Dataset source:
https://www.kaggle.com/imakash3011/customer-personality-analysis?select=marketing_campaign.csv
"""

import numpy as np
import pandas as pd
import joblib
from datetime import date
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

debug = False


def load_data(path):
    # Load data from file
    data = pd.read_csv(path, sep="\t")
    return data


def prepare(data, current_year, age_cap=90, income_cap=600000, training=False):
    if debug:
        mode = "training" if training else "prediction"
        print(f"\nPreparing data for {mode}")

    # Remove missing values
    data = data.dropna()

    # Create a feature out of "Dt_Customer" that indicates the number of
    # days a customer is registered in the firm's database
    data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"])
    max_date = data["Dt_Customer"].max()
    data["Customer_For"] = max_date
    data["Customer_For"] = (data["Customer_For"] - data["Dt_Customer"]).dt.days
    data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")

    # Age of customer as of today
    data["Age"] = current_year - data["Year_Birth"]

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
        data = data[data["Age"] < age_cap]
        data = data[data["Income"] < income_cap]

    if debug:
        print("Prepared data:\n", data)

    return data


def train(data, current_year, ncluster=4):
    data = prepare(data, current_year, training=True)

    #numerical = data.select_dtypes(include=["int64", "float64"]).columns
    categorical = data.select_dtypes(include=["object", "bool"]).columns
    if debug:
        print("Categorical variables in the dataset:", categorical.values)

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

    joblib.dump(pipeline, "model.joblib")


def predict(data, current_year):
    data = prepare(data, current_year, training=False)
    pipeline = joblib.load("model.joblib")
    predictions = pipeline.predict(data)
    data["Clusters"] = predictions
    if debug:
        print("Data with Predictions\n", data)
    return data


def main():
    np.random.seed(42)
    current_year = date.today().year
    data_path = "trainer/marketing_campaign.csv"
    data = load_data(data_path)
    train_data, test_data = train_test_split(data, test_size=0.10)
    train(train_data, current_year)
    predict(test_data.head(2), current_year)


if __name__ == "__main__":
    main()
