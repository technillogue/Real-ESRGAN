FROM python:3 as model
RUN pip install gdown
RUN gdown -O x4.pth 'https://drive.google.com/uc?id=1SGHdZAln4en65_NQeQY9UjchtkEF9f5F'

FROM ubuntu:hirsute as builder
WORKDIR /app
RUN echo "EST" > /etc/timezone
RUN apt update
RUN DEBIAN_FRONTEND="noninteractive" apt install -yy python3.9 python3.9-venv pipenv git
RUN python3.9 -m venv /app/venv
COPY Pipfile.lock Pipfile /app/
#ENV PIP_FIND_LINKS=https://download.pytorch.org/whl/cu113/torch_stable.html
ENV VIRTUALENV_PIP=22.0.3
RUN VIRTUAL_ENV=/app/venv pipenv install 

FROM ubuntu:hirsute
WORKDIR /app
RUN echo "EST" > /etc/timezone
RUN apt update && DEBIAN_FRONTEND="noninteractive" apt install -y python3.9
COPY --from=builder /app/venv/lib/python3.9/site-packages /app/
RUN mkdir inputs results weights
COPY --from=model /x4.pth /app/weights/
COPY ./arch_util.py ./postgres_jobs.py ./realesrgan.py ./rrdbnet_arch.py ./secrets.py ./utils_sr.py /app/ 
ENTRYPOINT ["/usr/bin/python3.9", "/app/postgres_jobs.py"]
