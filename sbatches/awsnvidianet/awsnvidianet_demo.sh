#!/bin/bash

#SBATCH -N 1                                   # number of nodes
#SBATCH -c 1                                   # number of "tasks" (cores)
#SBATCH --mem=4000                             # memory size per node (megabytes)

#SBATCH -t 0-00:30:00                          # time (DD-HH:MM:SS)

#SBATCH -p gpu                                 # partition type to use
#SBATCH -q wildfire                            # quality-of-service (QoS)

#SBATCH --gres=gpu:1                           # number of gpu's

#SBATCH -o awsnvidianet_test.out  # STDOUT
#SBATCH -e awsnvidianet_test.err  # STDERR

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
sed -i -E "0,/batch=/{s/batch=([0-9]+)/batch=1/}" awsnvidianet/cfg/awsnvidia.cfg
sed -i -E "0,/subdivisions=/{s/subdivisions=([0-9]+)/subdivisions=1/}" awsnvidianet/cfg/awsnvidia.cfg


# run darknet on example video
cd $HOME/darknet/
./darknet detector demo awsnvidianet/data/awsnvidia.data awsnvidianet/cfg/awsnvidia.cfg awsnvidianet/weights/awsnvidia.weights awsnvidianet/tests/asu-indoors-demo-1.mp4 -out_filename awsnvidianet/results/asu-indoors-demo-1-results.avi
