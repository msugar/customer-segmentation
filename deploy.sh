#!/bin/bash

# Set the current working directory to the directory of the script
cd "$(dirname "$0")"

PROJECT_ID=$(gcloud config get-value project)
REGION=us-central1 
REPOSITORY=custsegm-artifact-registry
IMAGE=custsegm

gcloud beta artifacts repositories create {REPOSITORY} \
 --repository-format=docker \
 --location="${REGION}"

docker push "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE}"

gcloud beta ai models upload \
  --region="${REGION}" \
  --display-name="${MODEL_NAME}" \
  --container-image-uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE}" \
  --container-ports=5050 \
  --container-health-route=/custsegm/v1 \
  --container-predict-route=/custsegm/v1/predict