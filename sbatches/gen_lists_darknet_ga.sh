#!/bin/bash

#SBATCH -N 1                                 # number of nodes
#SBATCH -c 1                                 # number of "tasks" (cores)
#SBATCH --mem=8000                           # memory size per node (megabytes)

#SBATCH -t 0-02:00:00                        # time (DD-HH:MM:SS)

#SBATCH -p serial                            # partition type to use
#SBATCH -q normal                            # quality-of-service (QoS)

#SBATCH -o outputs/gen_lists_darknet_ga.out  # STDOUT
#SBATCH -e outputs/gen_lists_darknet_ga.err  # STDERR

#SBATCH --mail-type=ALL                      # send email for start, stop, fail
#SBATCH --mail-user=jwande18@asu.edu         # send emails to

# purge existing modules
module purge

# load required modules
module load python/3.7.1

# enter Gaussian_YOLOv3 directory
cd /home/$USER/Gaussian_YOLOv3

# generate train list
sed -i -E "0,/path_data =/{s|path_data =.*|path_data = '/scratch/jwande18/OIDv4_ToolKit/OID/Dataset/train/images/'|}" generate_list.py
sed -i -E "0,/filename =/{s/filename =.*/filename = open('CSE485_train_list.txt', 'w')/}" generate_list.py
python3 generate_list.py

# generate test list
sed -i -E "0,/path_data =/{s|path_data =.*|path_data = '/scratch/jwande18/OIDv4_ToolKit/OID/Dataset/test/images/'|}" generate_list.py
sed -i -E "0,/filename =/{s/filename =.*/filename = open('CSE485_test_list.txt', 'w')/}" generate_list.py
python3 generate_list.py

# generate validation list
sed -i -E "0,/path_data =/{s|path_data =.*|path_data = '/scratch/jwande18/OIDv4_ToolKit/OID/Dataset/validation/images/'|}" generate_list.py
sed -i -E "0,/filename =/{s/filename =.*/filename = open('CSE485_validation_list.txt', 'w')/}" generate_list.py
python3 generate_list.py
