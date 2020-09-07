#!/bin/bash
DEFAULT_TAG="eu.gcr.io/neptune-cicd/neptune/docs:dev"
set -e

virtualenv -p python3.6 venv

source venv/bin/activate

pip install -r docs_requirements.txt

cd docs

make html

cd ../

docker build . -t "${1:-$DEFAULT_TAG}"
docker push ${1:-$DEFAULT_TAG}