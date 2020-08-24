#virtualenv -p python3 venv_blabla
#
#source venv_blabla/bin/activate
#
#pip install -r docs_requirements.txt

sphinx-apidoc -f -o docs/api-reference/neptune venv_blabla/lib/python3.8/site-packages/neptune
sphinx-apidoc -f -o docs/api-reference/neptune venv_blabla/lib/python3.8/site-packages/neptunecli
sphinx-apidoc -f -o docs/api-reference/neptunecontrib venv_blabla/lib/python3.8/site-packages/neptunecontrib
