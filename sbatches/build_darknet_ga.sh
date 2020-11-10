#!/bin/bash

#SBATCH -N 1                             # number of nodes
#SBATCH -c 1                             # number of "tasks" (cores)
#SBATCH --mem=4000                       # memory size per node (megabytes)

#SBATCH -t 0-01:00:00                    # time (DD-HH:MM:SS)

#SBATCH -p serial                        # partition type to use
#SBATCH -q normal                        # quality-of-service (QoS)

#SBATCH -o outputs/build_darknet_ga.out  # STDOUT
#SBATCH -e outputs/build_darknet_ga.err  # STDERR

#SBATCH --mail-type=ALL                  # send email for start, stop, fail
#SBATCH --mail-user=jwande18@asu.edu     # send emails to

# purge existing modules
module purge

# load required modules
module load cuda/10.2.89 \
            cudnn/7.1.3  \
            opencv/3.4.1 \
            make/4.1

# set global environment variables
export PKG_CONFIG_PATH="/packages/7x/opencv/3.4.1/lib64/pkgconfig/:$PKG_CONFIG_PATH"
export CPATH="/packages/7x/cudnn/7.0/include/:$CPATH"

# download darknet
cd $HOME
git clone https://www.github.com/jwchoi384/Gaussian_YOLOv3
cd Gaussian_YOLOv3

# compile darknet
make
