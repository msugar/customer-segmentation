# Customer Segmentation
Term Project for University of Toronto's "SCS 3760 004 Cloud Technologies for Big Data, Machine Learning &amp; Artificial Intelligence" course, Fall 2021. 

Keywords: Google Cloud Platform; GCP; Vertex AI Workbench; Docker; Custom Container; Cloud Run; Artifact Registry; Cloud Build; Cloud Logging.

## 1. Setup Google Cloud
* Create a new GCP project;
* Enable these APIs: Cloud Run, Cloud Build, Cloud Logging, and Vertex AI;
* Create a bucket named `gs://<PROJECT_ID>-bucket`, where `<PROJECT_ID>` is to be replaced with th ID of the GCP project;
* Copy the unzipped content of the [Customer Personality Analysis' dataset](https://www.kaggle.com/imakash3011/customer-personality-analysis?marketing_campaign.csv) to `gs://<PROJECT_ID>-bucket/custom-training/custsegm/data/marketing_campaign.csv`.

## 2. Setup your working environment (Linux)
* Clone the repo:
  ```
  git clone https://github.com/msugar/customer-segmentation.git
  cd customer_segmentation
  ```
* [Install and init the Google Cloud SDK](https://cloud.google.com/sdk/docs/install). _(Skip if you are in a Vertex AI Workbench's User-Managed Notebook.)_ 
* Whenever you start a new working session, create/update your Python 3.7+ virtual environment:
  ```
  source setup.sh
  ```

## 3. Build and deploy the service
* Build the docker image locally:
  ```
  ./docker-build.sh cloudrun
  ```
* Submit the built image to Cloud Build -> Artifactory Registry -> Cloud Run:
  ```
  ./docker-deploy.sh cloudrun
  ```
* Test the service's health:
  ```
  curl <SERVICE_URL>
  ```
* Make a couple of predictions:
  ```
  curl -X POST -H "Content-Type: application/json" <SERVICE_URL> -d "@input.json"
  ```
