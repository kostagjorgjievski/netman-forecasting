# Time-Series Forecasting Pipeline

This repository provides the complete experimental pipeline accompanying the paper **Comparative Analysis and Implementation of Many-to-One Time-Series Forecasting Models** It implements a unified framework for dataset preparation, training, evaluation, and analysis of three representative forecasting architectures, **DeepAR (GluonTS/PyTorch)**, **PatchTST**, and **iTransformer**, across the **ETT**, **Weather**, and **SMD** datasets.  

All auxiliary components, including training logic, evaluation scripts, logging, and AWS SageMaker orchestration, have been implemented from scratch. PatchTST and iTransformer are included as unmodified open-source baselines for reproducibility. The experiments benchmark all models under a consistent many-to-one forecasting protocol and four prediction horizons (96, 192, 336, 720), revealing complementary strengths: DeepAR performs best on smooth, high-frequency data; PatchTST captures local periodic structure effectively; and iTransformer generalizes well on high-dimensional, regime-shifting inputs. All preprocessing, hyperparameters, and evaluation procedures are fully implemented here to ensure one-to-one reproducibility of the paper’s reported results.

---

## Repository Structure

The codebase is organized into three logical layers that together form a modular and reproducible forecasting framework:

- **Command-Line Interface (CLI):** Located under `cli/`, these scripts control model training, evaluation, and AWS SageMaker job orchestration.  
- **Core Components:** Contained in `src/`, this layer includes dataset loaders, normalization utilities, model wrappers, metrics tracking, and experiment logging.  
- **Analysis Tools:** Found in `analysis/`, these scripts handle dataset characterization, exploratory analysis, and automated SageMaker Processing workflows.  

All experiment outputs are automatically written to the local `results/` directory or to an S3 bucket when executed via SageMaker.

---

## Environment and Installation

The framework has been validated on **Python 3.10–3.11** under Linux, macOS, and WSL2. GPU acceleration via PyTorch is supported but not required.  
Two isolated environments are recommended, one for **PatchTST / iTransformer**, and one for **DeepAR (GluonTS)**, to maintain dependency stability.

- `requirements.txt` - dependencies for PatchTST and iTransformer  
- `requirements-deepar.txt` - dependencies for DeepAR  

### Creating a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows users: .venv\Scripts\activate
python -m pip install --upgrade pip
```

### Installing Dependencies

```bash
# For PatchTST and iTransformer
pip install -r requirements.txt

# For DeepAR (GluonTS)
pip install -r requirements-deepar.txt

# For local dataset analysis
pip install matplotlib==3.7.2 statsmodels==0.14.1 scipy pandas
```

SageMaker automatically resolves dependencies at runtime, but using these environments locally ensures version consistency and replicable behavior.

---

## Datasets and Preprocessing

The pipeline accepts **numeric CSV files** where the final column is treated as the target variable. Non-numeric columns (e.g., timestamps) are ignored for training but retained for plotting or time indexing.  

Missing or infinite values are imputed via **causal forward-fill**, and leading NaNs are replaced with zeros. This process ensures smooth model input and consistent normalization across datasets.

Primary datasets include:

- **ETT:** Hourly industrial data with strong periodicity and smooth trend.  
- **Weather:** Ten-minute meteorological data with high-frequency seasonality.  
- **SMD:** Minute-level server telemetry data with irregular, regime-shifting behavior.  

This preprocessing design enforces structural uniformity across datasets while preserving their intrinsic temporal properties.

---

## Local Training and Evaluation

The unified training interface is accessible via `cli/train.py`, which supports all three model families. It automatically configures dataset-specific hyperparameters and forecasting horizons.

**Example (iTransformer):**

```bash
python cli/train.py   --model itransformer   --csv_path ./data/ETT.csv   --seq_len 336   --pred_len 96   --epochs 10   --batch_size 128   --logdir results
```

Replace `--model itransformer` with `--model patchtst` or `--model deepar` to train the respective architectures.  
All artifacts (checkpoints, logs, and per-epoch metrics) are stored under `results/`.

### DeepAR (GluonTS Implementation)

For probabilistic training using GluonTS’ native backend:

```bash
python cli/train_deepar_gluonts.py   --csv_path ./data/ETT.csv   --seq_len 336   --pred_len 96   --epochs 10   --batch_size 128   --freq h   --likelihood student_t   --scaling zscore   --logdir results
```

This configuration follows the canonical GluonTS workflow and outputs `metrics.json` and per-epoch logs for post-hoc evaluation.

### Dataset Analysis

The dataset analysis module (`analysis/spot_analyze.py`) computes descriptive statistics, zero and missing-value ratios, STL-based trend and seasonality strength, ADF/KPSS stationarity, and spectral forecastability.  

**Example:**

```bash
CSV_PATH=./data/ETT.csv DATASET_NAME=ett python analysis/spot_analyze.py
```

Outputs include CSV summaries and diagnostic visualizations (ACF/PACF, STL decompositions, etc.).

---

## Experimental Outputs

Each completed experiment produces a structured folder containing:

- `hparams.json` - configuration metadata  
- `epoch_log.csv` - epoch-level metrics  
- `summary.json` - best and final validation results  
- `figs/` - convergence, gradient, and loss plots  
- `preds/` - optional forecast samples  

A consolidated summary CSV and JSON are automatically generated at the project root to simplify comparative evaluation.

---

## AWS SageMaker Integration

The pipeline integrates seamlessly with **AWS SageMaker** for distributed, spot-optimized training and automated dataset analysis.

### Configuration and Permissions

Ensure the following environment variables are defined before launching jobs:

```bash
export AWS_REGION="us-west-2"
export S3_BUCKET="your-bucket-name"
export SAGEMAKER_ROLE_ARN="arn:aws:iam::<account-id>:role/<your-sagemaker-role>"
```

The SageMaker role must include permissions for:

- `s3:GetObject`, `s3:PutObject`, `s3:ListBucket`  
- `sagemaker:CreateTrainingJob`, `sagemaker:CreateProcessingJob`, `sagemaker:Describe*`  
- `logs:CreateLogStream`, `logs:PutLogEvents`  
- (If using ECR) `ecr:BatchGetImage`, `ecr:GetAuthorizationToken`

Your S3 directory structure should follow:

```
s3://$S3_BUCKET/code/
s3://$S3_BUCKET/data/
s3://$S3_BUCKET/outputs/
s3://$S3_BUCKET/checkpoints/
```

### Launching Training Jobs

The launcher `cli/run.py` automatically configures PyTorch estimators for each dataset and horizon.

**Example (PatchTST / iTransformer batch):**

```bash
for model in patchtst itransformer; do
  for dataset in ETT.csv weather.csv SMD.csv; do
    for H in 96 192 336 720; do
      python sm_jobs/launch_sm.py         --model $model         --dataset $dataset         --horizons $H         --epochs 30         --enc_in 21         --smd_granularity minute         --instance ml.g4dn.xlarge &
      sleep 1
    done
  done
done
```

Each job runs independently and streams logs to **CloudWatch**.  
A similar script can be used for **DeepAR**, substituting the `--model deepar` flag.

---

## Reproducibility Practices

All experiments were conducted under fixed random seeds across `random`, `numpy`, and `torch`.  
To enforce deterministic CUDA behavior:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

This configuration guarantees consistent validation and test metrics across repeated runs and hardware environments.

---

## Citations and Acknowledgments

The repository builds upon three key open-source forecasting frameworks:

- **PatchTST** - Nie, Y. et al. (2023). *A Time-Series is Worth 64 Words: Long-Term Forecasting with Transformers.*  
- **iTransformer** - Liu, C. et al. (2024). *iTransformer: Inverted Transformer for Time-Series Forecasting.*  
- **DeepAR** - Salinas, D. et al. (2019). *DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks.*

All remaining components, including dataset preprocessing, unified training loops, AWS integration, and analysis scripts, were developed specifically for this project.

---

## Troubleshooting Notes

- **Dependency conflicts:** Run `pip install "numpy<2.0"` if version errors occur.  
- **GPU unavailable:** Use `--device cpu` to force CPU execution.  
- **S3 permission errors:** Verify your IAM role includes `s3:GetObject` and `s3:PutObject`.  
- **Missing outputs:** Ensure `/opt/ml/processing/reports` and `/opt/ml/processing/figures` are correctly mapped.  
- **AWS environment issues:** Confirm that `AWS_REGION`, `S3_BUCKET`, and `SAGEMAKER_ROLE_ARN` are correctly exported before launch.

---

### Summary

This repository provides a complete and reproducible framework for **time-series forecasting research** on both local and cloud infrastructures.  
It combines state-of-the-art model implementations, robust experiment tracking, and automated SageMaker workflows to support transparent benchmarking, reproducible experimentation, and extension to future multivariate forecasting research.
