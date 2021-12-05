#!/bin/bash

# Set the current working directory to the directory of the script
cd "$(dirname "$0")"

# Name of this script
ME=`basename "$0"`

usage()
{
cat << EOF
Deploy the docker image for the predictor, trainer, or cloudrun containers.
Usage: $ME [-h] TARGET
Parameters:
    TARGET    Deploy target: predictor, trainer, or cloudrun.
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
        echo 'You must inform a valid build TARGET: cloudrun, predictor, or trainer.' >&2
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

MODEL_NAME="customer-segmentation-model" #YOUR_CHANGE
ENDPOINT_NAME="customer-segmentation-endpoint" #YOUR_CHANGE

#gcloud beta artifacts repositories create {REPO_NAME} \
# --repository-format=docker \
# --location="${REGION}"

case "$TARGET" in
    trainer|predictor)
        echo "Push ${TARGET} docker image to the Artifact Repository on Google Cloud:"
        echo "---"
        docker push "${IMAGE_URI}"
        echo "---"
        echo "Pushed ${IMAGE_URI}"
        ;;
    predictor)
        exit
        
        echo 'See if model already exists on Vertex AI:'
        gcloud beta ai models list \
          --region=$REGION \
          --filter=display_name=$MODEL_NAME

        echo 'Upload custom container with model trainer to Vertex AI:"
        echo "(It can take 10 to 15 minutes...)'
        echo `date`
        gcloud beta ai models upload \
          --region="${REGION}" \
          --display-name="${MODEL_NAME}" \
          --container-image-uri="${IMAGE_URI}" \
          --container-ports=5050 \
          --container-health-route=/healthz \
          --container-predict-route=/predict
        echo "Model trainer custom container uploaded!"  
        echo `date`
        
        read -p "Enter MODEL_ID to proceed: " MODEL_ID

        echo "Create endpoint on Vertex AI"
        gcloud beta ai endpoints create \
          --region=$REGION \
          --display-name=$ENDPOINT_NAME
        
        echo "Deploy model on endpoint on Vertex AI"
        read -p "Enter ENDPOINT_ID to proceed: " ENDPOINT_ID
        read -p "Enter DEPLOYED_MODEL_NAME to proceed: " DEPLOYED_MODEL_NAME
        
        gcloud beta ai endpoints deploy-model $ENDPOINT_ID \
          --region=$REGION \
          --model=$MODEL_ID \
          --display-name=$DEPLOYED_MODEL_NAME \
          --machine-type=n1-standard-2 \
          --min-replica-count=1 \
          --max-replica-count=2 \
          --traffic-split=0=100
        ;;
    cloudrun)
        # Submit docker image to Google Cloud Build
        cp Dockerfile.cloudrun Dockerfile
        echo "Submit image to Google Cloud Build:"
        #gcloud builds submit --tag "${IMAGE_URI}"
        rm Dockerfile
        
        # Submit docker container on Google Cloud Run
        echo "Deploy app with model on Google Cloud Run"
        echo "---"
        #gcloud run deploy --image "${IMAGE_URI}" --platform managed --region=$REGION 
        echo "---"
        CLOUDRUN_URL=$(gcloud run services describe custsegm-${TARGET} --platform managed --region ${REGION} --format 'value(status.url)')
        echo "Service ${CLOUDRUN_URL} <-- ${IMAGE_URI} running on Google Clour Run"
        echo ""
        echo "As an authenticated user, try these commands to test the web server:"
        echo "curl ${CLOUDRUN_URL}"
        echo "curl -X POST -H \"Content-Type: application/json\" ${CLOUDRUN_URL} -d \"@input.json\""
esac
echo "Done."