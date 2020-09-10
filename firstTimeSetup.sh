set -e

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

rm -rf venv
python3 -m venv venv
source venv/bin/activate

pip install setuptools_rust

python setup.py develop
