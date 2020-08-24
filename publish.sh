#!/bin/bash
DEFAULT_TAG="eu.gcr.io/neptune-cicd/neptune/docs:dev"
set -e

virtualenv -p python3 venv

source venv/bin/activate

pip install -r docs_requirements.txt

# Create API docs for client and contrib
sphinx-apidoc -f -o docs/api-reference/neptune venv/lib/python3.8/site-packages/neptune
sphinx-apidoc -f -o docs/api-reference/neptune venv/lib/python3.8/site-packages/neptunecli
sphinx-apidoc -f -o docs/api-reference/neptunecontrib venv/lib/python3.8/site-packages/neptunecontrib

cd docs

make html

cd ../

docker build . -t "${1:-$DEFAULT_TAG}"
docker push ${1:-$DEFAULT_TAG}