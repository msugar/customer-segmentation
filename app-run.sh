#!/bin/bash

# Set the current working directory to the directory of the script
cd "$(dirname "$0")"

PROJECT_ID=$(gcloud config get-value project)
#export CLOUD_ML_PROJECT_ID=${PROJECT_ID}
export AIP_MODEL_DIR="gs://${PROJECT_ID}-bucket/custom-training/custsegm/model"

echo "Running predictor web service"
echo
echo "Try these commands in another termainal to test the web server:"
echo "curl http://0.0.0.0:5050/healthz"
echo 'curl -X POST -H "Content-Type: application/json" http://0.0.0.0:5050/predict -d "@input.json"'
echo ""
gunicorn --bind 0.0.0.0:5050 --timeout=150 app.app:app -w 1
echo "Done"