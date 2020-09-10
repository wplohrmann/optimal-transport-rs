set -e

if [ "$( which cargo )" == "" ]; then
    PATH="$HOME/.cargo/bin:$PATH"
fi
source venv/bin/activate
python setup.py develop

python example.py
