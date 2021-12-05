#!/bin/bash

# Set the current working directory to the directory of the script
cd "$(dirname "$0")"

# Name of this script
ME=`basename "$0"`

usage()
{
cat << EOF
Runs a trainer or predictor docker container locally for testing.
Usage: $ME [-h] TARGET
Parameters:
    TARGET    Target image: predictor, trainer, or cloudrun.
Options:
    -h        Show this help.
EOF
}

if [ $# -eq 0 ]
  then
    usage
    exit 2
fi

# Process the input options
while getopts ":h" options; do
    case "${options}" in
        h)
            usage
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 2
            ;;
        :)
            echo "Option -$OPTARG requires an argument" >&2
            exit 2
    esac
done
shift $((OPTIND-1))

# Process the input parameters
unset TARGET
while [ $# -ne 0 ]
do
    arg="$1"
    case "$arg" in
        predictor|trainer|cloudrun)
            TARGET="$arg"
            ;;
        \?)
            echo "Invalid parameter $arg" >&2
            exit 2
            ;;
     esac
    shift
done

if [ -z "$TARGET" ]; then
        echo 'You must inform a valid build TARGET: predictor or trainer.' >&2
        exit 2
fi

echo "TARGET=$TARGET"

# Set docker image URI
PROJECT_ID=$(gcloud config get-value project)
REGION=us-central1 # YOUR_CHANGE
REPO_NAME=custsegm-docker-repo # YOUR_CHANGE the name of the Artifact Registry repository that you created
IMAGE_NAME="custsegm-${TARGET}" # YOUR_CHANGE a name of your choice for your container image
IMAGE_TAG=1.0 # YOUR_CHANGE a tag of your choice for this version of your container image
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

# Environment variables to pass to the container
#CLOUD_ML_PROJECT_ID="${PROJECT_ID}"
# YOUR_CHANGE use your own conventions for the default location of the input training dataset
AIP_TRAINING_DATA_URI="gs://${PROJECT_ID}-bucket/custom-training/custsegm/data/marketing_campaign.csv" 
# YOUR_CHANGE use your own conventions for the default location of the output model artifacts directory
AIP_MODEL_DIR="gs://${PROJECT_ID}-bucket/custom-training/custsegm/model"

case "$TARGET" in
    trainer)
        echo "Running ${TARGET} container locally in interactive mode to ensure it's working correctly."
        echo ""
        echo "IMAGE_URI=${IMAGE_URI}"
        echo ""
        docker run -it \
            -e AIP_TRAINING_DATA_URI="${AIP_TRAINING_DATA_URI}" \
            -e AIP_MODEL_DIR="${AIP_MODEL_DIR}" \
            --name=custsegm_trainer \
            "${IMAGE_URI}"
        docker rm "/custsegm_trainer"
        ;;
    predictor)
        echo "Running ${TARGET} container locally in interactive mode to ensure it's working correctly."
        echo ""
        echo "Try these commands in another termainal to test the web server:"
        echo "curl http://0.0.0.0:5050/healthz"
        echo 'curl -X POST -H "Content-Type: application/json" http://0.0.0.0:5050/predict -d "@input.json"'
        echo ""
        echo "IMAGE_URI=${IMAGE_URI}"
        echo ""
        docker run -it \
            -e AIP_MODEL_DIR="${AIP_MODEL_DIR}" \
            -p 5050:5050 \
            --name=custsegm_predictor \
            "${IMAGE_URI}"
        docker rm "/custsegm_predictor"
        ;;
    cloudrun)
        echo "Running ${TARGET} container locally in interactive mode to ensure it's working correctly."
        echo ""
        echo "Try these commands in another termainal to test the web server:"
        echo "curl http://0.0.0.0:5050"
        echo 'curl -X POST -H "Content-Type: application/json" http://0.0.0.0:5050 -d "@input.json"'
        echo ""
        echo "IMAGE_URI=${IMAGE_URI}"
        echo ""
        docker run -it \
            -e PORT=5050 \
            -p 5050:5050 \
            --name=custsegm_cloudrun \
            "${IMAGE_URI}"
        docker rm "/custsegm_cloudrun"
esac
echo "Done."