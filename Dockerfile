FROM python:3.10
WORKDIR /code 
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install -q git+https://github.com/huggingface/transformers.git
RUN pip install -q git+https://github.com/huggingface/peft.git


COPY . /code
EXPOSE 8001

HEALTHCHECK CMD curl --fail http://localhost:8001/_stcore/health

CMD ["streamlit", "run", "main.py", "--server.port=8001", "--server.address=0.0.0.0"]
