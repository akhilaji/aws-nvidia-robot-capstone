FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# prevent prompt from asking for timezone
ENV TZ=US/Arizona
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install essentials
RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        git \
        libgtk2.0-dev \
        pkg-config \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev

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

# install opencv
RUN git clone https://github.com/opencv/opencv.git workspace/git/opencv
RUN cd workspace/git/opencv && mkdir build && cd build && \
        cmake \
                -DWITH_QT=ON \ 
                -DWITH_OPENGL=ON \ 
                -DFORCE_VTK=ON \
                -DWITH_TBB=ON \
                -DWITH_GDAL=ON \
                -DWITH_XINE=ON \
                -DBUILD_EXAMPLES=ON \
                .. && \
        make -j4 && make install