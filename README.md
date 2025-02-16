# OmniParser SageMaker Deployment

This directory contains code for deploying OmniParser v2 to Amazon SageMaker as an asynchronous inference endpoint.

## Directory Structure

```
omniparser-sagemaker/
├── container/              # Container files for SageMaker deployment
│   ├── Dockerfile         # Docker configuration for the container
│   └── inference.py       # SageMaker model server implementation
├── model/                 # Model artifacts
│   ├── download_weights.py # Script to download weights from Hugging Face
│   └── weights/           # Local directory for temporary weight storage
├── scripts/               # Deployment and build scripts
│   ├── build_and_push.sh # Script to build and push Docker image to ECR
│   └── deploy.py         # Script to deploy model to SageMaker
└── examples/              # Example code for using the endpoint
    └── invoke_endpoint.py # Example of invoking the async endpoint
```

## Prerequisites

1. AWS CLI installed and configured with appropriate credentials
2. Docker installed and running
3. Python 3.12 or later
4. Required Python packages:
   ```
   boto3
   sagemaker
   Pillow
   huggingface-hub  # For downloading model weights
   ```

## Step-by-Step Deployment

### 1. Set Up Environment

```bash
# Install required packages
pip install -r requirements.txt

# Configure AWS CLI with your credentials
aws configure
```

### 2. Build and Push Container

```bash
cd sagemaker/scripts

# Set your S3 bucket for model weights
export OMNIPARSER_MODEL_BUCKET="your-model-bucket-name"

# Build and push (this will also download and upload model weights)
./build_and_push.sh
```

This script will:
- Create the S3 bucket if it doesn't exist
- Download model weights from Hugging Face
- Create a tarball and upload to: `s3://${OMNIPARSER_MODEL_BUCKET}/model/omniparser-v2/model.tar.gz`
- Build and push the Docker container to ECR

### 3. Deploy to SageMaker

```python
from scripts.deploy import deploy_omniparser

# Deploy using the same bucket used in build step
predictor = deploy_omniparser(
    model_bucket="your-model-bucket-name"
)
```

This will:
- Create a SageMaker model using the ECR container
- Configure the model to use weights from S3
- Deploy an async inference endpoint
- Return a predictor object for making inferences

### 4. Test the Endpoint

```python
from examples.invoke_endpoint import invoke_omniparser, get_results

# Submit an inference request
image_path = "path/to/your/image.png"
output_location = invoke_omniparser(image_path)

# Wait for processing (you can implement polling here)
import time
time.sleep(30)

# Get results
labeled_image, coordinates, content = get_results(output_location)
```

## Model Weights

OmniParser v2 uses two main model components:
1. Icon Detection Model (YOLO-based)
2. Icon Caption Model (Florence2)

The weights are managed in two stages:

1. **Build Time**:
   - Downloaded from Hugging Face
   - Packaged into `model.tar.gz`
   - Uploaded to S3: `s3://<bucket>/model/omniparser-v2/model.tar.gz`

2. **Runtime**:
   - SageMaker automatically downloads weights from S3
   - Extracts to `/opt/ml/model` in the container
   - Used by the model for inference

## Configuration

### Build Configuration
```bash
# Required:
export OMNIPARSER_MODEL_BUCKET="your-bucket"  # S3 bucket for model weights

# Optional:
export AWS_DEFAULT_REGION="us-west-2"  # Defaults to us-west-2
```

### Deployment Configuration
```python
# In deploy.py:
predictor = deploy_omniparser(
    model_bucket="your-bucket",
    model_prefix="model/omniparser-v2"  # Optional, defaults to this value
)
```

### Inference Configuration
```python
# In invoke_endpoint.py:
request = {
    'image': encode_image(image_path),
    'box_threshold': 0.05,     # Detection confidence threshold
    'iou_threshold': 0.7,      # Box overlap threshold
    'use_paddleocr': False,    # Whether to use PaddleOCR
    'batch_size': 128          # Batch size for caption generation
}
```

## Monitoring

1. **CloudWatch Metrics**:
   - Endpoint invocations
   - Model latency
   - GPU utilization

2. **CloudWatch Logs**:
   - Container logs
   - Inference errors

3. **S3 Monitoring**:
   - Async inference results
   - Failed inference requests

## Troubleshooting

1. **Build Issues**:
   - Check S3 bucket permissions
   - Verify Hugging Face access
   - Check Docker build logs
   - Ensure enough disk space for weights

2. **Deployment Issues**:
   - Verify IAM roles have necessary permissions
   - Check SageMaker service quotas
   - Verify GPU instance availability

3. **Inference Issues**:
   - Check async output location
   - Verify input image format
   - Monitor GPU memory usage

## Cleanup

```python
import boto3

# Delete endpoint
sagemaker = boto3.client('sagemaker')
sagemaker.delete_endpoint(EndpointName='omniparser-v2-async')

# Delete model weights (optional)
s3 = boto3.client('s3')
s3.delete_object(
    Bucket='your-model-bucket',
    Key='model/omniparser-v2/model.tar.gz'
)
``` 