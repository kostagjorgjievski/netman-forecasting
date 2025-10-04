from sagemaker.pytorch import pytorch

def make_estimator(role_arn, image_uri, bucket, instance_type = "ml.g5.2xlarge"):
    return PyTorch(
        entry_point = "cli/train.py",
        source_dir = ".", #ship your repo
        role = role_arn,
        image_uri = image_uri,
        instance_count = 1,
        instance_tryp = instance_type,
        use_spot_instances = True,
        max_run = 4 * 3600,
        max_wait = 6 * 3600
        checkpoint_s3_uri = f"s3://{bucket}/checkpoints",
        enable_sagemaker_metrics = True,
    )