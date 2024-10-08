# Pull base image
FROM ghcr.io/mlflow/mlflow:v2.11.1
EXPOSE 5000
EXPOSE 8000
COPY requirements.txt ./
COPY api.py ./
RUN mkdir ./temp
RUN chmod +777 -R ./temp
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "sh","-c","mlflow server --host 0.0.0.0 --port 5000 & fastapi dev api.py --host 0.0.0.0 --port 8000" ]

#CMD [ "mlflow","server","--host","0.0.0.0" ]
#not necassary for local infra
# mlflow server \
#     --backend-store-uri /mnt/persistent-disk \
#     --default-artifact-root s3://my-mlflow-bucket/ \
#     --host 0.0.0.0


#----How to build this image
#cmd:
#docker build -t mlflow_local .
#----How to run this image
#cmd:
#docker run -d -p 5000:5000 -p 8000:8000  mlflow_local