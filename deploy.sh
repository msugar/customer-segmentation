#!/bin/bash

# Set the current working directory to the directory of the script
cd "$(dirname "$0")"

# Set docker image URI
PROJECT_ID=$(gcloud config get-value project)
REGION=us-central1 # YOUR_CHANGE
REPO_NAME=custsegm-docker-repo # YOUR_CHANGE
IMAGE_NAME=custsegm # YOUR_CHANGE
IMAGE_TAG=1.0 # YOUR_CHANGE
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

MODEL_NAME="customer-segmentation" #YOUR_CHANGE

gcloud beta artifacts repositories create {REPO_NAME} \
 --repository-format=docker \
 --location="${REGION}"

docker push "${IMAGE_URI}"

gcloud beta ai models upload \
  --region="${REGION}" \
  --display-name="${MODEL_NAME}" \
  --container-image-uri="${IMAGE_URI}" \
  --container-ports=5050 \
  --container-health-route=/custsegm/v1 \
  --container-predict-route=/custsegm/v1/predict