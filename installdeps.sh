#/usr/bin/bash
/home/blast/.pyenv/versions/3.10.13/bin/python3 -m venv .venv
source .venv/bin/activate; pip3 install -r requirements.txt
source .venv/bin/activate; pip3 install --no-deps unsilence==1.0.9