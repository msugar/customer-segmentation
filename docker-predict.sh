#!/bin/bash

# Set the current working directory to the directory of the script
cd "$(dirname "$0")"

INPUT_DATA_FILE="input.json"

curl -X POST -H "Content-Type: application/json" http://0.0.0.0:5050/predict -d "@${INPUT_DATA_FILE}"
