#!/bin/bash
# Check if Pipenv is installed
pipenv --venv
pipenv update
pipenv install
pipenv shell
