set -e

PATH="$HOME/.cargo/bin:$PATH"
source venv/bin/activate
python setup.py develop

python example.py
