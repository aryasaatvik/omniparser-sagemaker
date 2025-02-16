"""
Script to download OmniParser model weights from Hugging Face and prepare for SageMaker deployment.
"""

import os
import shutil
import subprocess
from pathlib import Path
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelWeightsManager:
    """Manages downloading and preparing model weights for SageMaker deployment."""
    
    def __init__(self, target_dir: str):
        self.target_dir = Path(target_dir)
        self.model_files = {
            "icon_detect": [
                "train_args.yaml",
                "model.pt",
                "model.yaml"
            ],
            "icon_caption": [
                "config.json",
                "generation_config.json",
                "model.safetensors"
            ]
        }
    
    def download_from_huggingface(self):
        """Download model weights from Hugging Face."""
        logger.info("Downloading model weights from Hugging Face...")
        
        # Create target directory if it doesn't exist
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download files in a single command per model type
            for model_type, files in self.model_files.items():
                logger.info(f"Downloading {model_type} files...")
                
                cmd = f"cd {self.target_dir} && "
                for file in files:
                    file_path = f"{model_type}/{file}"
                    cmd += f"huggingface-cli download microsoft/OmniParser-v2.0 {file_path} --local-dir . && "
                cmd = cmd.rstrip(" && ")  # Remove trailing &&
                
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"Error downloading {model_type} files: {result.stderr}")
                    raise Exception(f"Failed to download {model_type} files")
            
            # Rename icon_caption to icon_caption_florence
            caption_dir = self.target_dir / "icon_caption"
            florence_dir = self.target_dir / "icon_caption_florence"
            if caption_dir.exists():
                logger.info("Renaming icon_caption to icon_caption_florence...")
                if florence_dir.exists():
                    shutil.rmtree(florence_dir)
                caption_dir.rename(florence_dir)
            
            logger.info("Successfully downloaded all model weights")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading model weights: {str(e)}")
            raise

def main():
    """Main function to download weights."""
    parser = argparse.ArgumentParser(description='Download OmniParser model weights')
    parser.add_argument('--target-dir', type=str, default='/opt/ml/model',
                      help='Directory to download weights to')
    args = parser.parse_args()
    
    manager = ModelWeightsManager(args.target_dir)
    
    try:
        # Download weights from Hugging Face
        manager.download_from_huggingface()
        
    except Exception as e:
        logger.error(f"Failed to download model weights: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 