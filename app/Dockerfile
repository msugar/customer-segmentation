FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-7

WORKDIR /

# Copies the trainer code to the docker image.
COPY custsegm /custsegm

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "custsegm.train"]
