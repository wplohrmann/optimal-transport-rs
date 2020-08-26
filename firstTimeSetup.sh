set -e

rm -rf venv
python3 -m venv venv
source venv/bin/activate

pip install setuptools_rust

python setup.py develop
