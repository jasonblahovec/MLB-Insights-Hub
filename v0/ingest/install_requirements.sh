#!/bin/bash

set -e

# Update and install pip
apt update
apt install -y python3-pip

# Install Python packages
pip3 install requests || { echo "Failed to install requests"; exit 1; }
pip3 install pandas || { echo "Failed to install pandas"; exit 1; }
pip3 install google-cloud-storage || { echo "Failed to install google-cloud-storage"; exit 1; }
pip3 install pyspark || { echo "Failed to install pyspark"; exit 1; }
pip3 install google-auth || { echo "Failed to install google-auth"; exit 1; }
pip3 install google-cloud-secret-manager || { echo "Failed to install google-cloud-secret-manager"; exit 1; }
pip3 install python-mlb-statsapi || { echo "Failed to install python-mlb-statsapi"; exit 1; }
pip3 install jupyter || { echo "Failed to install jupyter"; exit 1; }
