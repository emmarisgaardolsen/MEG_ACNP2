#!/bin/bash

# create virtual envrionment
python3 -m venv virt_env

# activate virtual environment
source ./env/bin/activate

# install requirements
python3 -m pip install -r requirements.txt

# deactivate env
deactivate