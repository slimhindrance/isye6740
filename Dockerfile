FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip &&\
	r -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /home/lindeman/test/isye6740

COPY . /opt/test

RUN pip install --no-cache-dir --upgrade pip --break-system-packages \
	&& pip install --no-cache-dir -r requirements.txt --break-system-packages

EXPOSE 8000

CMD["uvicorN", "app:app", "--host", "0.0.0.0", "--port", "8000"]
