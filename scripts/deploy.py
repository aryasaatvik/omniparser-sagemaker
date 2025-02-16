import boto3
import sagemaker
from sagemaker.async_inference.async_inference_config import AsyncInferenceConfig
from sagemaker.model import Model
from sagemaker.predictor import Predictor
import os
import time

def cleanup_sagemaker_resources(sagemaker_client, endpoint_name):
    """Clean up existing SageMaker resources."""
    try:
        # Delete endpoint
        print(f"Deleting endpoint {endpoint_name}...")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        waiter = sagemaker_client.get_waiter('endpoint_deleted')
        waiter.wait(EndpointName=endpoint_name)
    except sagemaker_client.exceptions.ClientError as e:
        if "Could not find endpoint" not in str(e):
            raise

    try:
        # Delete endpoint config
        print(f"Deleting endpoint configuration {endpoint_name}...")
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
    except sagemaker_client.exceptions.ClientError as e:
        if "Could not find endpoint configuration" not in str(e):
            raise

    try:
        # Delete model
        print(f"Deleting model omniparser-v2...")
        sagemaker_client.delete_model(ModelName="omniparser-v2")
    except sagemaker_client.exceptions.ClientError as e:
        if "Could not find model" not in str(e):
            raise

def deploy_omniparser(model_bucket, model_prefix="model/omniparser-v2"):
    """Deploy OmniParser model to SageMaker endpoint.
    
    Args:
        model_bucket (str): S3 bucket containing model artifacts
        model_prefix (str): S3 prefix where model.tar.gz is located
        
    Returns:
        sagemaker.predictor.Predictor: Predictor object for the endpoint
        
    Raises:
        ValueError: If required environment variables are missing
        Exception: If deployment fails
    """
    try:
        # Initialize SageMaker session
        sagemaker_session = sagemaker.Session()
        account = boto3.client('sts').get_caller_identity()['Account']
        region = boto3.session.Session().region_name or 'us-west-2'
        
        # Build the ECR image URI

        image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/omniparser:latest"
        
        # Verify model artifacts exist
        s3_client = boto3.client('s3')
        try:
            s3_client.head_object(
                Bucket=model_bucket,
                Key=f"{model_prefix}/model.tar.gz"
            )
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise ValueError(
                    f"Model artifacts not found at s3://{model_bucket}/{model_prefix}/model.tar.gz"
                )
            raise
        
        # Model artifacts S3 location
        model_data = f"s3://{model_bucket}/{model_prefix}/model.tar.gz"
        
        # Configure async inference
        async_config = AsyncInferenceConfig(
            output_path=f"s3://{sagemaker_session.default_bucket()}/omniparser/async-output",
            max_concurrent_invocations_per_instance=2
        )

        # Clean up existing resources
        endpoint_name = "omniparser-v2-async"
        cleanup_sagemaker_resources(sagemaker_session.sagemaker_client, endpoint_name)
        
        # Create SageMaker model
        model = Model(
            image_uri=image_uri,
            model_data=model_data,
            role=f"arn:aws:iam::{account}:role/OmniParserSageMakerRole",
            predictor_cls=Predictor,
            name="omniparser-v2",
            env={
                'OMNIPARSER_MODEL_BUCKET': model_bucket,
                'MODEL_PATH': '/opt/ml/model'
            }
        )
        
        # Deploy as async endpoint
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type="ml.g4dn.xlarge",  # GPU instance
            endpoint_name=endpoint_name,
            async_inference_config=async_config,
            wait=True
        )
        
        print(f"Endpoint deployed: {predictor.endpoint_name}")
        return predictor
        
    except Exception as e:
        print(f"Error deploying model: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    model_bucket = os.environ.get('OMNIPARSER_MODEL_BUCKET')
    if not model_bucket:
        raise ValueError("OMNIPARSER_MODEL_BUCKET environment variable must be set")
    deploy_omniparser(model_bucket) 