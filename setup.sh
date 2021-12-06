#!/bin/bash

# Make sure this script is sourced, not run
if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "Hey, you should source this script, not execute it!"
    exit 1
fi

# Set environment variables
PROJECT_ID=$(gcloud config get-value project)
export AIP_TRAINING_DATA_URI="gs://${PROJECT_ID}-bucket/custom-training/custsegm/data/marketing_campaign.csv"
export AIP_MODEL_DIR="gs://${PROJECT_ID}-bucket/custom-training/custsegm/model"

# Check gsutil exists, which indicates Cloud SDK is installed.
if ! gsutil -v &> /dev/null
then
    echo "ERROR: gsutil could not be found. Have you installed the Google Cloud SDK?"
else
    # Check the dataset file exists in the expected location on GCS for this project
    gsutil -q stat ${AIP_TRAINING_DATA_URI}
    status=$?
    if [[ ! $status == 0 ]]; then
        echo "ERROR: Training dataset not found on GCP project '${PROJECT_ID}' !"
        echo "       Make sure a bucket with the expected name exists and is accessible:"
        echo "         gs://${PROJECT_ID}-bucket"
        echo "       Make sure you have the dataset copied to the expected location:"
        echo "         ${AIP_TRAINING_DATA_URI}"
        echo "       The original dataset can be found in Kaggle:"
        echo "         https://www.kaggle.com/imakash3011/customer-personality-analysis?marketing_campaign.csv"
    else
        # Setup virtual enviroment
        if [ ! -d "./venv" ] 
        then
            echo 'Setting up virtual environment (venv)' 
            python -m venv venv
        fi
        echo 'Activating virtual environment (venv)' 
        source ./venv/bin/activate
        echo '(Re)installing dependencies' 
        pip --disable-pip-version-check install -r requirements.txt --quiet
        echo "Setup successful. Environment is ready for use."
    fi
fi




