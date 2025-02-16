"""
OmniParser handler for serving inference requests using SageMaker multi-model server.
"""
import os
import json
import torch
import base64
import io
from PIL import Image
from typing import List, Dict, Any

from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img


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
        self.initialized = True
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        # Set device
        gpu_id = properties.get("gpu_id", 0)
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        
        # Load models
        try:
            self.yolo_model = get_yolo_model(
                model_path=os.path.join(model_dir, 'icon_detect/model.pt')
            )
            self.caption_model_processor = get_caption_model_processor(
                model_name="florence2",
                model_name_or_path=os.path.join(model_dir, 'icon_caption_florence')
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {str(e)}")

    def preprocess(self, request: List[Dict[str, Any]]) -> Dict:
        """
        Transform raw input into model input data.
        
        Args:
            request: List of raw requests
            
        Returns:
            Dict containing preprocessed model input data
        """
        if not request or len(request) != 1:
            raise ValueError("Only single request batch is supported")
            
        # Get request data
        data = request[0]
        if not isinstance(data, dict):
            data = json.loads(data.get('body').decode('utf-8'))
            
        # Get image data
        image_data = data.get('image')
        if not image_data:
            raise ValueError("No image data provided in request")
            
        # Convert base64 to PIL Image
        if isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        elif isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        else:
            raise ValueError("Unsupported image format")
            
        # Get inference parameters with defaults
        return {
            'box_threshold': data.get('box_threshold', 0.05),
            'iou_threshold': data.get('iou_threshold', 0.7),
            'use_paddleocr': data.get('use_paddleocr', True),
            'imgsz': data.get('imgsz', 640),
            'image': image
        }

    def inference(self, model_input: Dict) -> Dict:
        """
        Internal inference methods.
        
        Args:
            model_input: Transformed model input data
            
        Returns:
            Dict containing inference results
        """
        image = model_input['image']
        
        # Save image temporarily
        tmp_path = '/tmp/input_image.png'
        image.save(tmp_path)
        
        try:
            # Configure box overlay
            box_overlay_ratio = image.size[0] / 3200
            draw_bbox_config = {
                'text_scale': 0.8 * box_overlay_ratio,
                'text_thickness': max(int(2 * box_overlay_ratio), 1),
                'text_padding': max(int(3 * box_overlay_ratio), 1),
                'thickness': max(int(3 * box_overlay_ratio), 1),
            }
            
            # Run OCR
            ocr_bbox_rslt, _ = check_ocr_box(
                tmp_path,
                display_img=False,
                output_bb_format='xyxy',
                goal_filtering=None,
                easyocr_args={'paragraph': False, 'text_threshold': 0.9},
                use_paddleocr=model_input['use_paddleocr']
            )
            text, ocr_bbox = ocr_bbox_rslt
            
            # Run inference
            labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
                tmp_path,
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
            
            return {
                'labeled_image': labeled_img,  # This is already base64 encoded
                'coordinates': label_coordinates,
                'parsed_content': parsed_content_list
            }
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def postprocess(self, inference_output: Dict) -> List[Dict]:
        """
        Return prediction result in list format.
        
        Args:
            inference_output: Raw inference output
            
        Returns:
            List containing processed prediction results
        """
        return [{
            'labeled_image': inference_output['labeled_image'],
            'coordinates': inference_output['coordinates'],
            'parsed_content': [
                f'icon {i}: {content}'
                for i, content in enumerate(inference_output['parsed_content'])
            ]
        }]

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions.
        
        Args:
            data: Input data
            context: MMS context
        """
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        return self.postprocess(model_out)


_service = ModelHandler()


def handle(data, context):
    """
    Top-level handler function.
    
    Args:
        data: Input data
        context: MMS context
    """
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context) 