#!/bin/bash

# Set the current working directory to the directory of the script
cd "$(dirname "$0")"

# Name of this script
ME=`basename "$0"`

usage()
{
cat << EOF
Build a docker image for the trainer or predictor container.
Usage: $ME [-h] TARGET
Parameters:
    TARGET    Build target: predictor or trainer.
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
        predictor|trainer)
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

# Build docker image
echo "Building ${TARGET} docker image: ${IMAGE_URI}"
docker build --file="Dockerfile.${TARGET}" --tag="${IMAGE_URI}" .

echo "Done"