# Specifies base image and tag
# A few options of Vertex AI pre-built containers can be found here:
# https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-7

WORKDIR /work

# Installs additional packages
#COPY ./requirements.txt /work/requirements.txt
#RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Copies the trainer code to the docker image.
COPY ./custsegm /work/custsegm

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "custsegm.trainer"]