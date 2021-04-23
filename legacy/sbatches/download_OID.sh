#!/bin/bash

#SBATCH -N 1                             # number of nodes
#SBATCH -c 8                             # number of "tasks" (cores)
#SBATCH --mem=8000                       # memory size per node (megabytes)

#SBATCH -t 0-24:00:00                    # time (DD-HH:MM:SS)

#SBATCH -p serial                        # partition type to use
#SBATCH -q normal                        # quality-of-service (QoS)

#SBATCH -o outputs/download_OID.out  # STDOUT
#SBATCH -e outputs/download_OID.err  # STDERR

#SBATCH --mail-type=ALL                  # send email for start, stop, fail
#SBATCH --mail-user=jwande18@asu.edu     # send emails to

# purge existing modules
module purge

# load required modules
module load git/2.8.0 \
            python/3.7.1 

# download OIDv4 toolkit
cd /scratch/$USER
git clone https://github.com/EscVM/OIDv4_ToolKit.git

# install requirements
cd /scratch/$USER/OIDv4_ToolKit/
pip3 install --user pip==18.1
pip install --user  -r requirements.txt

# edit downloader.py
sed -i -E "0,/rows/{s/rows.*/rows, columns = [170, 30]/}" modules/downloader.py

# set classes
export OID_CLASSES="
Bottle \
Chair \
Computer_keyboard \
Computer_monitor \
Computer_mouse \
Corded_phone \
Desk \
Headphones \
Laptop \
Microphone \
Mobile_phone \
Mug \
Office_building \
Office_supplies \
Pen \
Person \
Stapler \
Table \
Tablet_computer \
Telephone \
Television \
Whiteboard"

# download train images
python3 main.py downloader \
--classes $OID_CLASSES \
--type_csv train \
--multiclasses 1 \
--limit 1500 \
--n_threads 16 -y

# download test dataset
python3 main.py downloader \
--classes $OID_CLASSES \
--type_csv test \
--multiclasses 1 \
--limit 1500 \
--n_threads 16 -y

# download validation dataset
python3 main.py downloader \
--classes $OID_CLASSES \
--type_csv validation \
--multiclasses 1 \
--limit 1500 \
--n_threads 16 -y
