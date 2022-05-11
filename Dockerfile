FROM amazon/aws-lambda-nodejs
RUN yum install tar git wget openssl-devel python-pip -y \
    && yum groupinstall "Development Tools" -y \
    && pip install cmake --upgrade
RUN git clone --depth 1 https://github.com/tensorflow/tensorflow.git tensorflow_src
RUN mkdir tflite_build
WORKDIR tflite_build
RUN sed -i -e 's/common.c/common.cc/g' ../tensorflow_src/tensorflow/lite/c/CMakeLists.txt
RUN cmake ../tensorflow_src/tensorflow/lite/c
RUN cmake --build . -j 2

# docker cp extract-tflite-c:/var/task/tflite_build/libtensorflowlite_c.so .

RUN yum install python3 -y

COPY package.json ./
RUN npm install
RUN cp /var/task/tflite_build/libtensorflowlite_c.so /var/task/tflite_build/node_modules/node-tflite/build/Release/libtensorflowlite_c.so

COPY api ./api/
COPY static ./static/
RUN node api/embedder.js