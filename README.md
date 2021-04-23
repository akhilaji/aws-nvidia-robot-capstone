# AWS-Nvidia Robot for Image Difference Detection

_Fall 2020 - Spring 2021_

Akhil Aji, Jacob Anderson, Jacob Cohen, Preston Mott, Jacob Schmit, Riley Tuoti

**Sponsors**: Dr. Yinong Chen, Dr. Gennaro De Luca

The AWS-Nvidia Robot for Image Difference Detection is an autonomous robotic system capable of navigating an office space and reporting changes in a scene.

**Prerequisites**    
Python3, pip   
Install CUDA with the CUDnn toolkit if using a compatible device.
installation steps can be found here:
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html   
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html    

**Repository Setup**
*using requirements.txt*

`pip3 install -r requirements.txt`

pull model weights and accompanying files from:    
https://drive.google.com/drive/folders/1wjuxzDXU_JTpQkCxrCgGGo-E8slB0HLY?usp=sharing

paste these folders into /src/integration    

**Tensorflow Setup**

https://www.tensorflow.org/install/gpu

The Custom YoloV4 Model can detect objects from the following classes:  
Bottle   
Chair  
Computer keyboard  
Computer monitor  
Computer mouse  
Corded phone  
Desk  
Headphones  
Laptop  
Microphone  
Mobile phone  
Mug  
Office supplies  
Pen  
Person  
Stapler  
Table  
Telephone  
Television  
Whiteboard  
Filing cabinet  
Picture frame  

## Functionality



​    



