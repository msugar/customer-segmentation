#!/bin/bash

# Set the current working directory to the directory of the script
cd "$(dirname "$0")"

# Set docker image URI
PROJECT_ID=$(gcloud config get-value project)
REGION=us-central1 # YOUR_CHANGE
REPO_NAME=custsegm-docker-repo # YOUR_CHANGE the name of the Artifact Registry repository that you created
IMAGE_NAME=custsegm # YOUR_CHANGE a name of your choice for your container image
IMAGE_TAG=1.0 # YOUR_CHANGE a tag of your choice for this version of your container image
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

# Login to docker, if you are not logged in
#docker login

# Build docker image
echo "Building docker image: ${IMAGE_URI}"
docker build ./ --tag="${IMAGE_URI}" .

# Run container locally as a basic test
echo "Smoke test: Run the container locally to ensure it's working correctly"
#docker run -d -p 5050:5050 --name=test-container "${IMAGE_URI}"
# YOUR_CHANGE default locations of input training dataset file and output model artifacts directory
docker run \
    -e AIP_TRAINING_DATA_URI="gs://${PROJECT_ID}-bucket/custom-training/custsegm/data/marketing_campaign.csv" \
    -e AIP_MODEL_DIR="gs://${PROJECT_ID}-bucket/custom-training/custsegm/model" \
    "${IMAGE_URI}"