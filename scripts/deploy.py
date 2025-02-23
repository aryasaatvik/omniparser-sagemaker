import boto3
import sagemaker
from sagemaker.async_inference.async_inference_config import AsyncInferenceConfig
from sagemaker.model import Model
from sagemaker.predictor import Predictor
import os
import time
import argparse

def create_sns_topics(region: str, account: str) -> tuple:
    """Create SNS topics for async inference notifications if they don't exist.
    
    Args:
        region (str): AWS region
        account (str): AWS account ID
        
    Returns:
        tuple: (success_topic_arn, error_topic_arn)
    """
    print("Setting up SNS topics for async inference notifications...")
    sns = boto3.client('sns')
    
    # Create success topic
    success_topic_name = "omniparser-success"
    success_topic_arn = f"arn:aws:sns:{region}:{account}:{success_topic_name}"
    try:
        sns.get_topic_attributes(TopicArn=success_topic_arn)
        print(f"Success topic {success_topic_name} already exists")
    except sns.exceptions.NotFoundException:
        print(f"Creating success topic {success_topic_name}...")
        response = sns.create_topic(Name=success_topic_name)
        success_topic_arn = response['TopicArn']
        
        # Add tags
        sns.tag_resource(
            ResourceArn=success_topic_arn,
            Tags=[
                {'Key': 'Purpose', 'Value': 'SageMaker-Async-Inference'},
                {'Key': 'Service', 'Value': 'OmniParser'}
            ]
        )
    
    # Create error topic
    error_topic_name = "omniparser-error"
    error_topic_arn = f"arn:aws:sns:{region}:{account}:{error_topic_name}"
    try:
        sns.get_topic_attributes(TopicArn=error_topic_arn)
        print(f"Error topic {error_topic_name} already exists")
    except sns.exceptions.NotFoundException:
        print(f"Creating error topic {error_topic_name}...")
        response = sns.create_topic(Name=error_topic_name)
        error_topic_arn = response['TopicArn']
        
        # Add tags
        sns.tag_resource(
            ResourceArn=error_topic_arn,
            Tags=[
                {'Key': 'Purpose', 'Value': 'SageMaker-Async-Inference'},
                {'Key': 'Service', 'Value': 'OmniParser'}
            ]
        )
    
    print("SNS topics setup complete")
    return success_topic_arn, error_topic_arn

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

def setup_autoscaling(endpoint_name: str, min_capacity: int = 0, max_capacity: int = 2):
    """Set up autoscaling for the endpoint based on queue size.
    
    Args:
        endpoint_name (str): Name of the SageMaker endpoint
        min_capacity (int): Minimum number of instances (default: 0)
        max_capacity (int): Maximum number of instances (default: 2)
    """
    print(f"Setting up autoscaling for endpoint {endpoint_name}...")
    application_autoscaling = boto3.client('application-autoscaling')
    
    # Register the endpoint as a scalable target
    application_autoscaling.register_scalable_target(
        ServiceNamespace='sagemaker',
        ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        MinCapacity=min_capacity,
        MaxCapacity=max_capacity
    )
    
    # Configure scaling policy based on queue size
    application_autoscaling.put_scaling_policy(
        PolicyName=f'{endpoint_name}-queue-based-autoscaling',
        ServiceNamespace='sagemaker',
        ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        PolicyType='TargetTrackingScaling',
        TargetTrackingScalingPolicyConfiguration={
            'TargetValue': 5.0,  # Target 5 requests in queue per instance
            'CustomizedMetricSpecification': {
                'MetricName': 'ApproximateBacklogSizePerInstance',
                'Namespace': 'AWS/SageMaker',
                'Dimensions': [
                    {'Name': 'EndpointName', 'Value': endpoint_name},
                    {'Name': 'VariantName', 'Value': 'AllTraffic'}
                ],
                'Statistic': 'Average'
            },
            'ScaleInCooldown': 600,  # 10 minutes
            'ScaleOutCooldown': 300   # 5 minutes
        }
    )
    print("Autoscaling configured successfully")

def deploy_omniparser(model_bucket, model_prefix="model/omniparser-v2", delete_only=False):
    """Deploy OmniParser model to SageMaker endpoint.
    
    Args:
        model_bucket (str): S3 bucket containing model artifacts
        model_prefix (str): S3 prefix where model.tar.gz is located
        delete_only (bool): If True, only delete existing resources without deploying
        
    Returns:
        sagemaker.predictor.Predictor: Predictor object for the endpoint, None if delete_only=True
        
    Raises:
        ValueError: If required environment variables are missing
        Exception: If deployment fails
    """
    try:
        # Initialize SageMaker session
        sagemaker_session = sagemaker.Session()
        endpoint_name = "omniparser-v2-async"
        
        # Clean up existing resources
        print("Cleaning up existing resources...")
        cleanup_sagemaker_resources(sagemaker_session.sagemaker_client, endpoint_name)
        
        if delete_only:
            print("Delete-only mode, skipping deployment")
            return None
            
        account = boto3.client('sts').get_caller_identity()['Account']
        region = boto3.session.Session().region_name or 'us-west-2'
        
        # Create SNS topics if they don't exist
        success_topic_arn, error_topic_arn = create_sns_topics(region, account)
        
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
        
        # Configure async inference with SNS notifications
        async_config = AsyncInferenceConfig(
            output_path=f"s3://{sagemaker_session.default_bucket()}/omniparser/async-output",
            max_concurrent_invocations_per_instance=2,
            notification_config={
                "SuccessTopic": success_topic_arn,
                "ErrorTopic": error_topic_arn,
                "IncludeInferenceResponseIn": ["SUCCESS_NOTIFICATION_TOPIC", "ERROR_NOTIFICATION_TOPIC"]
            }
        )
        
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
            instance_type="ml.g6.xlarge",  # GPU instance
            endpoint_name=endpoint_name,
            async_inference_config=async_config,
            wait=True
        )
        
        print(f"Endpoint deployed: {predictor.endpoint_name}")
        
        # Set up autoscaling
        setup_autoscaling(
            endpoint_name=endpoint_name,
            min_capacity=0,  # Scale to zero when no requests
            max_capacity=2   # Max 2 instances
        )
        
        return predictor
        
    except Exception as e:
        print(f"Error deploying model: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deploy or delete OmniParser SageMaker endpoint')
    parser.add_argument('--model-bucket', type=str, help='S3 bucket containing model artifacts',
                      default=os.environ.get('OMNIPARSER_MODEL_BUCKET'))
    parser.add_argument('--model-prefix', type=str, default="model/omniparser-v2",
                      help='S3 prefix where model.tar.gz is located')
    parser.add_argument('--delete-only', action='store_true',
                      help='Only delete existing resources without deploying')
    
    args = parser.parse_args()
    
    if not args.model_bucket and not args.delete_only:
        raise ValueError("OMNIPARSER_MODEL_BUCKET environment variable or --model-bucket argument must be set")
    
    deploy_omniparser(
        model_bucket=args.model_bucket,
        model_prefix=args.model_prefix,
        delete_only=args.delete_only
    ) 