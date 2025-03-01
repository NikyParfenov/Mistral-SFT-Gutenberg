FROM python:3.10

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /code 
COPY ./requirements.txt /code/requirements.txt
# COPY ./download_models.py /code/download_models.py

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install -q git+https://github.com/huggingface/transformers.git
RUN pip install -q git+https://github.com/huggingface/peft.git
RUN curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu18.04/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
# RUN python3 /code/download_models.py

COPY . /code
EXPOSE 8001

HEALTHCHECK CMD curl --fail http://localhost:8001/_stcore/health

CMD ["streamlit", "run", "main.py", "--server.port=8001", "--server.address=0.0.0.0"]
