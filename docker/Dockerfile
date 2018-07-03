# build the image
# cd $DETECTRON/docker
# docker build -t detectron:c2-cuda9-cudnn7 .

# run the image nvidia-docker run --rm -it detectron:c2-cuda9-cudnn7 python2 detectron/tests/test_batch_permutation_op.py

#Log in to virtual server using
#ssh -L 8888:localhost:8888 ubuntu@<your ip address from setup email>

# Pull and start the container:
# nvidia-docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow:latest-gpu

# Then open http://localhost:8888 in your browser.
#You can use the option "-v /ssdtemp:/ssdtemp" to map the 800 GB of storage space mounten on /ssdtemp in the server to the container.


# Use Caffe2 image as parent image
FROM caffe2/caffe2:snapshot-py2-cuda9.0-cudnn7-ubuntu16.04

RUN mv /usr/local/caffe2 /usr/local/caffe2_build
ENV Caffe2_DIR /usr/local/caffe2_build

ENV PYTHONPATH /usr/local/caffe2_build:${PYTHONPATH}
ENV LD_LIBRARY_PATH /usr/local/caffe2_build/lib:${LD_LIBRARY_PATH}

# Clone the Detectron repository
# use own version of detectrion
RUN ping github.com -c 4
RUN git clone https://github.com/StadlerMaximilian/dataset_setup.git ./test
RUN git clone https://github.com/StadlerMaximilian/Detectron /detectron_docker

# Install Python dependencies
RUN pip install -r /detectron_docker/requirements.txt

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
WORKDIR /cocoapi/PythonAPI
RUN make install

# Go to Detectron root
WORKDIR /detectron_docker

# Set up Python modules
RUN make

# [Optional] Build custom ops
RUN make ops
