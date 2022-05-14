FROM amazon/aws-lambda-nodejs
RUN npm install -g pnpm
RUN yum install python3 -y
RUN yum install -y make gcc*

COPY package.json ./
COPY static/libtensorflowlite_c.so libtensorflowlite_c.so
RUN npm install

COPY api ./api/
COPY static ./static/

CMD ["sh", "-c", "tail -f /dev/null"]
