#!/usr/bin/env/bash
virtualenv -p python3 --system-site-packages ../image_detection
ln -s ../image_detection
sudo ./image_detection/bin/pip3 install -r config/requirements.txt

