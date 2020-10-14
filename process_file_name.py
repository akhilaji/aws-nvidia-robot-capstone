import os
from pathlib import Path
print("hello")

data_dir = Path("./images/")
df_path  = Path("./annotations/trainval.txt")
with open(df_path, "a") as data_file:
    for file in os.listdir(data_dir):
        if file.endswith(".jpg"):
            print(os.path.splitext(file)[0])
            data_file.write(os.path.splitext(file)[0] + "\n")
        else:
            continue