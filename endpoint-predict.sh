#!/bin/bash

# Set the current working directory to the directory of the script
cd "$(dirname "$0")"

# Set docker image URI
PROJECT_ID=$(gcloud config get-value project)
ENDPOINT_ID="1294652951473684480" # YOUR_CHANGE the endpoint id
INPUT_DATA_FILE="input.json"

curl \
-X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/us-central1/endpoints/${ENDPOINT_ID}:predict \
-d "@${INPUT_DATA_FILE}"
