import boto3
import json
import time
from PIL import Image
import io
import os
from botocore.exceptions import ClientError

def upload_test_image(image_path, bucket, prefix="omniparser/test-images"):
    """Upload test image to S3 and return its URI.
    
    Args:
        image_path (str): Path to the image file
        bucket (str): S3 bucket name
        prefix (str): S3 prefix for test images
        
    Returns:
        str: S3 URI of the uploaded image
    """
    s3 = boto3.client('s3')
    
    # Generate S3 key
    filename = os.path.basename(image_path)
    timestamp = int(time.time())
    key = f"{prefix}/{timestamp}/{filename}"
    
    # Upload image
    print(f"Uploading test image to s3://{bucket}/{key}")
    with open(image_path, 'rb') as f:
        s3.upload_fileobj(f, bucket, key)
    
    return f"s3://{bucket}/{key}"

def test_async_endpoint(image_path, endpoint_name="omniparser-v2-async"):
    """Test the async inference endpoint with an image.
    
    Args:
        image_path (str): Path to the image file
        endpoint_name (str): Name of the SageMaker endpoint
    """
    # Initialize clients
    runtime = boto3.client('sagemaker-runtime')
    s3 = boto3.client('s3')
    sagemaker = boto3.client('sagemaker')
    
    # Get default bucket
    account = boto3.client('sts').get_caller_identity()['Account']
    region = boto3.session.Session().region_name
    default_bucket = f"sagemaker-{region}-{account}"
    
    try:
        # Upload test image to S3
        image_uri = upload_test_image(image_path, default_bucket)
        
        # Prepare the input
        input_data = {
            "image_uri": image_uri,
            "box_threshold": 0.05,
            "iou_threshold": 0.7,
            "use_paddleocr": True,
            "imgsz": 640
        }
        
        # Upload input to S3
        print("Uploading input to S3...")
        input_key = f"omniparser/async-input/{int(time.time())}/input.json"
        s3.put_object(
            Bucket=default_bucket,
            Key=input_key,
            Body=json.dumps(input_data)
        )
        input_location = f"s3://{default_bucket}/{input_key}"
        print(f"Input uploaded to: {input_location}")
        
        # Send async inference request
        print("Sending async inference request...")
        response = runtime.invoke_endpoint_async(
            EndpointName=endpoint_name,
            ContentType='application/json',
            InputLocation=input_location
        )
        
        # Get the output location
        output_location = response['OutputLocation']
        print(f"Request submitted. Output will be available at: {output_location}")
        
        # Parse S3 location
        s3_parts = output_location.replace("s3://", "").split("/")
        bucket = s3_parts[0]
        key = "/".join(s3_parts[1:])
        
        # Poll for results
        print("Waiting for results...")
        max_tries = 30
        tries = 0
        while tries < max_tries:
            try:
                response = s3.get_object(Bucket=bucket, Key=key)
                print("Results received!")
                break
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    time.sleep(10)
                    tries += 1
                    print(f"Still processing... (attempt {tries}/{max_tries})")
                else:
                    raise
        
        if tries == max_tries:
            print("Timeout waiting for results")
            return
        
        # Parse results
        results = json.loads(response['Body'].read().decode('utf-8'))
        result = results[0]
        
        # Print results
        print("\nResults:")
        print(f"Annotated image: {result['image_uri']}")
        print("\nParsed Content:")
        for content in result.get('parsed_content', []):
            print(content)
        print("\nCoordinates:")
        print(json.dumps(result.get('coordinates', {}), indent=2))
        
        # Cleanup
        print("\nCleaning up input files from S3...")
        s3.delete_object(Bucket=default_bucket, Key=input_key)
        
        # Note: We don't delete the input or output images as they might be needed for reference
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test OmniParser async endpoint')
    parser.add_argument('image_path', help='Path to image file to process')
    parser.add_argument('--endpoint-name', default='omniparser-v2-async',
                      help='SageMaker endpoint name')
    
    args = parser.parse_args()
    test_async_endpoint(args.image_path, args.endpoint_name) 