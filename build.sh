#!/bin/bash

# Set the current working directory to the directory of the script
cd "$(dirname "$0")"

PROJECT_ID=$(gcloud config get-value project)
REGION=us-central1 
REPOSITORY=custsegm-artifact-registry
IMAGE=custsegm

# Login to docker, if you are not logged in
docker login

# Build docker image
docker build --tag="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE}" .

# Test Custom Container Locally
docker run -d -p 5050:5050 --name=custsegm_container "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE}"