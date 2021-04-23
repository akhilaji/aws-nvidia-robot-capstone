#!/bin/bash

#SBATCH -N 1                                   # number of nodes
#SBATCH -c 1                                   # number of "tasks" (cores)
#SBATCH --mem=4000                             # memory size per node (megabytes)

#SBATCH -t 0-00:30:00                          # time (DD-HH:MM:SS)

#SBATCH -p gpu                                 # partition type to use
#SBATCH -q wildfire                            # quality-of-service (QoS)

#SBATCH --gres=gpu:1                           # number of gpu's

#SBATCH -o outputs/test_CSE485_darknet_ga.out  # STDOUT
#SBATCH -e outputs/test_CSE485_darknet_ga.err  # STDERR

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

# set batch and subdivisions to 1
sed -i -E "0,/batch=/{s/batch=([0-9]+)/batch=1/}" cfg/Gaussian_yolov3_CSE485.cfg
sed -i -E "0,/subdivisions=/{s/subdivisions=([0-9]+)/subdivisions=1/}" cfg/Gaussian_yolov3_CSE485.cfg

# run darknet on example video
cd $HOME/Gaussian_YOLOv3/
./darknet detector demo cfg/CSE485.data cfg/Gaussian_yolov3_CSE485.cfg backup/Gaussian_yolov3_CSE485_30000.weights data/TwoWomenWalkingInOfficeOriginal.mp4 -out_filename results.avi
