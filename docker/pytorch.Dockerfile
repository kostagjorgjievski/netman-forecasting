# GPU base (use CPU tag if you prefer)
FROM public.ecr.aws/deeplearning/pytorch-training:2.2.0-gpu-py310-cu118-ubuntu20.04

# Add your deps + SM training entrypoint
RUN pip install --no-cache-dir numpy pandas einops torchmetrics sagemaker-training
COPY . /opt/ml/code
WORKDIR /opt/ml/code
ENV SAGEMAKER_PROGRAM cli/train.py
