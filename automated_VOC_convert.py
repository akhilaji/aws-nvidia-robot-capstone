#Convert annotation files
import os
import os.path
from pathlib import Path
from subprocess import check_output
currdir = os.getcwd()
print(currdir)
source =Path('./data/Dataset/train')
script_path = Path( './OIDv4_to_VOC-master/OIDv4_to_VOC.py')
#source = os.path.join("data", "Dataset", "train")

for folder in os.listdir(source):
    target = f'{source}/{folder}'
    output = check_output(["python", script_path , "--sourcepath" , f"{source}/{folder}", "--dest_path", f"{target}"])   