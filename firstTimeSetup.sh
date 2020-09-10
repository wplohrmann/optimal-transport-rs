set -e

if [ "$( which cargo )" == "" ]; then
    sudo curl -s https://sh.rustup.rs -sSf -o rustupinstaller.sh
    sh rustupinstaller.sh -y
    sudo rm rustupinstaller.sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

rm -rf venv
python3 -m venv venv
source venv/bin/activate

pip install setuptools_rust

python setup.py develop
