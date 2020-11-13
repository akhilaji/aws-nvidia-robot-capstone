# Arizona-State-University---AWS-NVidia-Robot-for-Image-Difference-Detection

**Repository Setup**

*without install.sh*

This will install requirements in a virtualenv. 

*using requirements.txt*

`pip3 install -r requirements.txt`


**Tensorflow Setup**

https://www.tensorflow.org/install/gpu

**Darknet Setup**  
Make sure compatible versions of CUDA and CUDNN are installed on your system 
`git clone https://github.com/AlexeyAB/darknet`

cd into the downloaded darknet repo and modify the makefile to enable openCV, CUDA, CUDNN. 
```
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
```
Make Darknet with command:
`make`

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
Office building
Office supplies
Pen
Person
Stapler
Table
Tablet computer
Telephone
Television
Whiteboard
