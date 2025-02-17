"""
OmniParser handler for serving inference requests using SageMaker multi-model server.
"""
import os
import json
import torch
import base64
import io
import logging
import sys
import traceback
from PIL import Image
from typing import List, Dict, Any
import boto3
from urllib.parse import urlparse

from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

# Configure logging to write to stdout
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def parse_s3_uri(uri: str) -> tuple:
    """Parse S3 URI into bucket and key.
    
    Args:
        uri (str): S3 URI in format s3://bucket/key
        
    Returns:
        tuple: (bucket, key)
    """
    parsed = urlparse(uri)
    if parsed.scheme != 's3':
        raise ValueError(f"Invalid S3 URI scheme: {parsed.scheme}")
    return parsed.netloc, parsed.path.lstrip('/')

def download_from_s3(uri: str) -> Image.Image:
    """Download image from S3 and return as PIL Image.
    
    Args:
        uri (str): S3 URI to image
        
    Returns:
        Image.Image: PIL Image object
    """
    bucket, key = parse_s3_uri(uri)
    s3_client = boto3.client('s3')
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_bytes = response['Body'].read()
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        logger.error(f"Failed to download image from {uri}: {str(e)}")
        raise

def upload_to_s3(image: Image.Image, uri: str, format: str = 'PNG') -> None:
    """Upload PIL Image to S3.
    
    Args:
        image (Image.Image): PIL Image to upload
        uri (str): Destination S3 URI
        format (str): Image format (default: PNG)
    """
    bucket, key = parse_s3_uri(uri)
    s3_client = boto3.client('s3')
    
    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    img_byte_arr = img_byte_arr.getvalue()
    
    try:
        s3_client.put_object(Bucket=bucket, Key=key, Body=img_byte_arr)
    except Exception as e:
        logger.error(f"Failed to upload image to {uri}: {str(e)}")
        raise

class ModelHandler(object):
    """
    Handler for OmniParser model serving.
    """
    def __init__(self):
        self.initialized = False
        self.device = None
        self.yolo_model = None
        self.caption_model_processor = None

    def initialize(self, context):
        """
        Initialize model. Called during model loading.
        
        Args:
            context: Initial context containing model server system properties.
        """
        logger.info("Initializing model handler...")
        self.initialized = True
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        logger.info(f"Model directory: {model_dir}")
        
        # Set device
        gpu_id = properties.get("gpu_id", 0)
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load models
        try:
            logger.info("Loading YOLO model...")
            self.yolo_model = get_yolo_model(
                model_path=os.path.join(model_dir, 'icon_detect/model.pt')
            )
            logger.info("Loading caption model and processor...")
            self.caption_model_processor = get_caption_model_processor(
                model_name="florence2",
                model_name_or_path=os.path.join(model_dir, 'icon_caption_florence')
            )
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to load models: {str(e)}")

    def preprocess(self, request: List[Dict[str, Any]]) -> Dict:
        """
        Transform raw input into model input data.
        
        Args:
            request: List of raw requests
            
        Returns:
            Dict containing preprocessed model input data
        """
        logger.info("Starting preprocessing...")
        logger.info(f"Request type: {type(request)}")
        
        if not request or len(request) != 1:
            logger.error("Invalid request batch size")
            raise ValueError("Only single request batch is supported")
            
        # Get request data
        data = request[0]
        logger.info(f"Request[0] type: {type(data)}")
        logger.info(f"Request[0] keys: {data.keys() if isinstance(data, dict) else 'not a dict'}")
        
        try:
            # Handle the case where we get a dict with 'body' key containing bytearray
            if isinstance(data, dict) and 'body' in data:
                body = data['body']
                logger.info(f"Body type: {type(body)}")
                if isinstance(body, bytearray):
                    body_str = body.decode('utf-8')
                    logger.info(f"Decoded body (first 100 chars): {body_str[:100]}...")
                    data = json.loads(body_str)
                    logger.info(f"Parsed body keys: {data.keys()}")
            
            # Get image URI
            image_uri = data.get('image_uri')
            if not image_uri:
                logger.error("No image_uri in request")
                logger.error(f"Available keys in data: {data.keys()}")
                raise ValueError("No image_uri provided in request")
                
            # Download image from S3
            logger.info(f"Downloading image from {image_uri}")
            image = download_from_s3(image_uri)
            logger.info(f"Image loaded successfully. Size: {image.size}, Mode: {image.mode}")
            
            # Get inference parameters with defaults
            params = {
                'box_threshold': data.get('box_threshold', 0.05),
                'iou_threshold': data.get('iou_threshold', 0.7),
                'use_paddleocr': data.get('use_paddleocr', True),
                'imgsz': data.get('imgsz', 640),
                'image': image,
                'image_uri': image_uri  # Pass through for postprocessing
            }
            logger.info(f"Preprocessing complete. Parameters: {str({k:v for k,v in params.items() if k not in ['image']})}")
            return params
            
        except Exception as e:
            logger.error(f"Failed to process request: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.error(f"Request data: {str(data)[:1000]}...")  # Log first 1000 chars of data
            raise

    def inference(self, model_input: Dict) -> Dict:
        """
        Internal inference methods.
        
        Args:
            model_input: Transformed model input data
            
        Returns:
            Dict containing inference results
        """
        logger.info("Starting inference...")
        image = model_input['image']
        
        try:
            # Configure box overlay
            box_overlay_ratio = image.size[0] / 3200
            draw_bbox_config = {
                'text_scale': 0.8 * box_overlay_ratio,
                'text_thickness': max(int(2 * box_overlay_ratio), 1),
                'text_padding': max(int(3 * box_overlay_ratio), 1),
                'thickness': max(int(3 * box_overlay_ratio), 1),
            }
            logger.info(f"Box overlay config: {draw_bbox_config}")
            
            # Run OCR
            logger.info("Running OCR...")
            ocr_bbox_rslt, _ = check_ocr_box(
                image,
                display_img=False,
                output_bb_format='xyxy',
                goal_filtering=None,
                easyocr_args={'paragraph': False, 'text_threshold': 0.9},
                use_paddleocr=model_input['use_paddleocr']
            )
            text, ocr_bbox = ocr_bbox_rslt
            logger.info(f"OCR complete. Found {len(text)} text elements")
            
            # Run inference
            logger.info("Running icon detection and captioning...")
            labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
                image,
                self.yolo_model,
                BOX_TRESHOLD=model_input['box_threshold'],
                output_coord_in_ratio=True,
                ocr_bbox=ocr_bbox,
                draw_bbox_config=draw_bbox_config,
                caption_model_processor=self.caption_model_processor,
                ocr_text=text,
                iou_threshold=model_input['iou_threshold'],
                imgsz=model_input['imgsz']
            )
            logger.info(f"Inference complete. Found {len(parsed_content_list)} icons")
            
            # Generate output URI by appending '_annotated' before extension
            input_uri = model_input['image_uri']
            base, ext = os.path.splitext(input_uri)
            output_uri = f"{base}_annotated{ext}"
            
            # Convert labeled_img from base64 to PIL Image
            if isinstance(labeled_img, str):
                img_bytes = base64.b64decode(labeled_img)
                labeled_img = Image.open(io.BytesIO(img_bytes))
            
            # Upload annotated image to S3
            logger.info(f"Uploading annotated image to {output_uri}")
            upload_to_s3(labeled_img, output_uri)
            
            return {
                'image_uri': output_uri,
                'coordinates': label_coordinates,
                'parsed_content': parsed_content_list
            }
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def postprocess(self, inference_output: Dict) -> List[Dict]:
        """
        Return prediction result in list format.
        
        Args:
            inference_output: Raw inference output
            
        Returns:
            List containing processed prediction results
        """
        logger.info("Starting postprocessing...")
        try:
            result = [{
                'image_uri': inference_output['image_uri'],
                'coordinates': inference_output['coordinates'],
                'parsed_content': [
                    f'icon {i}: {content}'
                    for i, content in enumerate(inference_output['parsed_content'])
                ]
            }]
            logger.info("Postprocessing complete")
            return result
        except Exception as e:
            logger.error(f"Error during postprocessing: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions.
        
        Args:
            data: Input data
            context: MMS context
        """
        logger.info("Starting request handling")
        if not _service.initialized:
            logger.info("Initializing service")
            _service.initialize(context)

        if data is None:
            logger.warning("Received null data")
            return None

        try:
            model_input = self.preprocess(data)
            model_out = self.inference(model_input)
            result = self.postprocess(model_out)
            logger.info("Request handling complete")
            return result
        except Exception as e:
            logger.error(f"Error handling request: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise


_service = ModelHandler()


def handle(data, context):
    """
    Top-level handler function.
    
    Args:
        data: Input data
        context: MMS context
    """
    return _service.handle(data, context) 