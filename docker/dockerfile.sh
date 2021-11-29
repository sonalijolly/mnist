FROM ubuntu:18.04
copy mnist /exp/mnist
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r /exp/mnist/requirements.txt
WORKDIR /exp
CMD ["python3", "./mnist/api/hello.py"]
