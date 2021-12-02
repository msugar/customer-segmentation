FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-7

WORKDIR /work

# Copies the trainer code to the docker image.
COPY . /work

RUN pip install --trusted-host pypi.python.org -r requirements.txt 

CMD gunicorn --bind 0.0.0.0:5005 --timeout=150 app:app -w 5

EXPOSE 5005