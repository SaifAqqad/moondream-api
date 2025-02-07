FROM python:3.13

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./torch-gpu-requirements.txt /code/torch-gpu-requirements.txt
# COPY ./torch-cpu-requirements.txt /code/torch-cpu-requirements.txt

# RUN pip install --no-cache-dir --upgrade -r /code/torch-cpu-requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/torch-gpu-requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

ENV USE_GPU true
CMD ["fastapi", "run", "app/main.py", "--port", "8000"]