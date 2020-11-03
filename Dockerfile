FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# prevent prompt from asking for timezone
ENV TZ=US/Arizona
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install essentials
RUN apt-get update && apt-get install -y \
        build-essential \
        libssl-dev \
        libgtk2.0-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libopencv-dev \
        pkg-config \
        wget \
        git

# install python
RUN apt-get update && apt-get install -y \
        python-dev \
        python-pip \
        python3-dev \
        python3-pip \
        python-numpy \
        python-pycurl \
        python-opencv

# creates workspace directory to act as a root for this project
RUN mkdir workspace && mkdir workspace/git

#install cmake
RUN cd workspace/git && \
        wget https://github.com/Kitware/CMake/releases/download/v3.16.5/cmake-3.16.5.tar.gz && \
        tar -zxvf cmake-3.16.5.tar.gz && rm cmake-3.16.5.tar.gz && \
        cd cmake-3.16.5 && sh bootstrap && \
        make -j"$(nproc)" && make install

# install tensorflow
RUN pip3 install tensorflow

# install opencv
#RUN git clone https://github.com/opencv/opencv.git workspace/git/opencv
#RUN cd workspace/git/opencv && mkdir build && cd build && \
#        cmake \
#                -DWITH_OPENGL=ON \ 
#                -DFORCE_VTK=ON \
#                -DWITH_TBB=ON \
#                -DWITH_GDAL=ON \
#                -DWITH_XINE=ON \
#                .. && \
#        make -j"$(nproc)" && make install

# install darknet
RUN git clone https://github.com/pjreddie/darknet.git workspace/git/darknet
RUN cd workspace/git/darknet && \
        sed -i 's/GPU=0/GPU=1/' Makefile && \
        sed -i 's/CUDNN=0/CUDNN=1/' Makefile && \
        sed -i 's/OPENCV=0/OPENCV=1/' Makefile && \
        make -j"$(nproc)"
