# aws/sagemaker_estimator.py
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import sagemaker, boto3, time

region = "us-east-1"
bucket = "netman-forecasting-329599632202"
sm_session = sagemaker.Session(boto_session=boto3.Session(region_name=region))

est = PyTorch(
    entry_point="train.py",
    source_dir="cli",                 # requirements.txt lives here
    dependencies=["src"],             # your package code
    role="arn:aws:iam::329599632202:role/netman-sagemaker-exec",
    framework_version="2.4",
    py_version="py311",
    instance_type="ml.m5.xlarge",     # CPU until GPU quota
    instance_count=1,
    sagemaker_session=sm_session,
    output_path=f"s3://{bucket}/outputs/",
    hyperparameters={
        "model": "itransformer",
        "csv_path": "/opt/ml/input/data/training/sample.csv",
        "seq_len": 96, "pred_len": 96, "epochs": 1,
    },
)

inputs = {"training": TrainingInput(s3_data=f"s3://{bucket}/datasets/", input_mode="File")}
est.fit(inputs=inputs, wait=True, logs="All")


