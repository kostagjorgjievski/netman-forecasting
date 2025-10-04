FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.3.0-gpu-py310-cu121-ubuntu20.04
RUN pip install --no-cache-dir numpy pandas einops torchmetrics
COPY . /opt/ml/code
WORKDIR /opt/ml/code
ENV SAGEMAKER_PROGRAM cli/train.py
