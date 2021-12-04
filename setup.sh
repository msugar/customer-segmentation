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

# Setup virtual enviroment
if [ ! -d "./venv" ] 
then
    echo 'Setting up virtual environment (venv)' 
    python -m venv venv
    source ./venv/bin/activate
    pip install -r requirements.txt
else
    echo 'Activating virtual environment (venv)' 
    source ./venv/bin/activate
fi


