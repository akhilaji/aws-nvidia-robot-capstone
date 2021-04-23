#!/bin/bash

#SBATCH -N 1                             # number of nodes
#SBATCH -c 1                             # number of "tasks" (cores)
#SBATCH --mem=8000                       # memory size per node (megabytes)

#SBATCH -t 0-01:00:00                    # time (DD-HH:MM:SS)

#SBATCH -p serial                        # partition type to use
#SBATCH -q normal                        # quality-of-service (QoS)

#SBATCH -o outputs/convert_OID.out       # STDOUT
#SBATCH -e outputs/convert_OID.err       # STDERR

#SBATCH --mail-type=ALL                  # send email for start, stop, fail
#SBATCH --mail-user=jwande18@asu.edu     # send emails to

# purge existing modules
module purge

# load required modules
module load git/2.8.0 \
            python/3.7.1 

# download OIDv4 toolkit
cd /scratch/$USER
git clone https://github.com/ibaiGorordo/OpenImages-Yolo-converter.git

# install modules
pip3 install --user pandas

# copy converter to csv_folder
cd /scratch/$USER/OpenImages-Yolo-converter/
cp OIDtoYOLOconverter.py /scratch/$USER/OIDv4_ToolKit/OID/csv_folder/OIDtoYOLOconverter.py
cd /scratch/$USER/OIDv4_ToolKit/OID/csv_folder/

# set classes to generate labels for
export OID_CLASSES='"Bottle", "Chair", "Computer keyboard", "Computer monitor", "Computer mouse", "Corded phone", "Desk", "Headphones", "Laptop", "Microphone", "Mobile phone", "Mug", "Office building", "Office supplies", "Pen", "Person", "Stapler", "Table", "Tablet computer", "Telephone", "Television", "Whiteboard"'
sed -i -E "0,/trainable_classes/{s/trainable_classes.*/trainable_classes = [${OID_CLASSES}]/}" OIDtoYOLOconverter.py

# fix pandas bug
sed -i -E "0,/class_descriptions/{s/class_descriptions.*/class_descriptions = pd.read_csv('class-descriptions-boxable.csv', names=['class_code','class_name'], header=None)/}" OIDtoYOLOconverter.py

# convert train labels
sed -i -E "0,/IMAGE_DIR/{s|IMAGE_DIR.*|IMAGE_DIR = '/scratch/jwande18/OIDv4_ToolKit/OID/Dataset/train/images/'|}" OIDtoYOLOconverter.py
sed -i -E "0,/annotation_files =/{s/annotation_files =.*/annotation_files = ['train-annotations-bbox.csv']/}" OIDtoYOLOconverter.py
sed -i -E "0,/filename =/{s/filename =.*/filename = 'train-annotations-bbox.csv'/}" OIDtoYOLOconverter.py
python3 OIDtoYOLOconverter.py

# convert test labels
sed -i -E "0,/IMAGE_DIR/{s|IMAGE_DIR.*|IMAGE_DIR = '/scratch/jwande18/OIDv4_ToolKit/OID/Dataset/test/images/'|}" OIDtoYOLOconverter.py
sed -i -E "0,/annotation_files =/{s/annotation_files =.*/annotation_files = ['test-annotations-bbox.csv']/}" OIDtoYOLOconverter.py
sed -i -E "0,/filename =/{s/filename =.*/filename = 'test-annotations-bbox.csv'/}" OIDtoYOLOconverter.py
python3 OIDtoYOLOconverter.py

# convert validation labels
sed -i -E "0,/IMAGE_DIR/{s|IMAGE_DIR.*|IMAGE_DIR = '/scratch/jwande18/OIDv4_ToolKit/OID/Dataset/validation/images/'|}" OIDtoYOLOconverter.py
sed -i -E "0,/annotation_files =/{s/annotation_files =.*/annotation_files = ['validation-annotations-bbox.csv']/}" OIDtoYOLOconverter.py
sed -i -E "0,/filename =/{s/filename =.*/filename = 'validation-annotations-bbox.csv'/}" OIDtoYOLOconverter.py
python3 OIDtoYOLOconverter.py
