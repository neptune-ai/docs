#!/bin/bash
DEFAULT_TAG="docker-repo.deepsense.codilime.com/deepsense/neptune/docs:dev"
set -e

virtualenv -p python3 venv

source venv/bin/activate

pip install -r docs_requirements.txt

cd docs

make html

cd ../

docker build . -t "${1:-$DEFAULT_TAG}"
docker push ${1:-$DEFAULT_TAG}