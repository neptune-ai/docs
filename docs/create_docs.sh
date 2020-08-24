rm -rf _build/html
make html
cd _build/html
python3 -m http.server