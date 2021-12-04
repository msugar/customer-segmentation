# Specifies base image and tag
# A few options of Vertex AI pre-built containers can be found here:
# https://cloud.google.com/vertex-ai/docs/training/pre-built-containers
# https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers
#FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-7

#FROM us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-7:latest
#FROM us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-7:latest

#FROM us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest
#FROM us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-23:latest
#FROM us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest

FROM python:3.7-slim

WORKDIR /work

# Installs additional packages
COPY ./requirements.txt /work/requirements.txt
RUN pip install -r requirements.txt
#RUN pip install -U --no-cache-dir --trusted-host pypi.python.org -r requirements.txt
#RUN pip install -U -r requirements.txt
#RUN pip3 install -U -r requirements.txt

# Copies the trainer code to the docker image.
COPY ./custsegm /work/custsegm

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "custsegm.trainer"]
#ENTRYPOINT ["python3", "-m", "custsegm.trainer"]