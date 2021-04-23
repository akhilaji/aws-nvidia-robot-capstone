#!/bin/bash

#SBATCH -N 1                            # number of nodes
#SBATCH -c 1                            # number of "tasks" (cores)
#SBATCH --mem=4000                      # memory size per node (megabytes)

#SBATCH -t 0-00:30:00                   # time (DD-HH:MM:SS)

#SBATCH -p gpu                          # partition type to use
#SBATCH -q wildfire                     # quality-of-service (QoS)

#SBATCH --gres=gpu:1                    # number of gpu's

#SBATCH -o outputs/test_darknet_ga.out  # STDOUT
#SBATCH -e outputs/test_darknet_ga.err  # STDERR

#purge existing modules
module purge

# load required modules
module load cuda/10.2.89 \
            cudnn/7.1.3  \
            opencv/3.4.1

# set global environment variables
export PKG_CONFIG_PATH="/packages/7x/opencv/3.4.1/lib64/pkgconfig/:$PKG_CONFIG_PATH"
export CPATH="/packages/7x/cudnn/7.0/include/:$CPATH"
export LD_LIBRARY_PATH="/packages/7x/opencv/3.4.1/lib64/:$LD_LIBRARY_PATH"

# download the BDD example weights
cd $HOME/Gaussian_YOLOv3/
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1Eutnens-3z6o4LYe0PZXJ1VYNwcZ6-2Y' -O- | sed -rn 's/.*confirm=([0-9A-Za-z]+).*/\1\n/p')&id=1Eutnens-3z6o4LYe0PZXJ1VYNwcZ6-2Y" -O Gaussian_yolov3_BDD.weights && rm -rf /tmp/cookies.txt

# set batch and subdivisions to 1
sed -i -E "0,/batch=/{s/batch=([0-9]+)/batch=1/}" cfg/Gaussian_yolov3_BDD.cfg
sed -i -E "0,/subdivisions=/{s/subdivisions=([0-9]+)/subdivisions=1/}" cfg/Gaussian_yolov3_BDD.cfg

# run darknet on example
./darknet detector test cfg/BDD.data cfg/Gaussian_yolov3_BDD.cfg Gaussian_yolov3_BDD.weights data/example.jpg

# e-mail the results
mailx -a predictions.jpg -s "Gaussian YOLOv3 Test Results" jwande18@asu.edu < /dev/null
