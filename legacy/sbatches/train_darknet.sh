#!/bin/bash

#SBATCH -N 1                             # number of nodes
#SBATCH -c 1                             # number of "tasks" (cores)
#SBATCH --mem=16000                      # memory size per node (megabytes)

#SBATCH -t 2-00:00:00                    # time (DD-HH:MM:SS)

#SBATCH -p gpu                           # partition type to use
#SBATCH -q wildfire                      # quality-of-service (QoS)
#SBATCH -C V100_16                       # requested gpu product
#SBATCH --gres=gpu:2                     # number of gpu's

#SBATCH -o outputs/train_darknet.out     # STDOUT
#SBATCH -e outputs/train_darknet.err     # STDERR

#SBATCH --mail-type=ALL                  # send email for start, stop, fail
#SBATCH --mail-user=jwande18@asu.edu     # send emails to

# purge existing modules
module purge

# load required modules
module load cuda/10.2.89 \
            cudnn/7.1.3  \
            opencv/3.4.1

export PKG_CONFIG_PATH="/packages/7x/opencv/3.4.1/lib64/pkgconfig/:$PGK_CONFIG_PATH"
export CPATH="/packages/7x/cudnn/7.0/include:$CPATH"

# Add opencv libraries to library path
export LD_LIBRARY_PATH="/packages/7x/opencv/3.4.1/lib64/:$LD_LIBRARY_PATH"

# train the gaussian model
cd /home/$USER/Gaussian_YOLOv3/
./darknet detector train cfg/CSE485.data cfg/Gaussian_yolov3_CSE485.cfg darknet53.conv.74 -gpus 0,1
