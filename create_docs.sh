rm -rf docs/_build/html
cd docs
make html
cd _build/html
python3 -m http.server