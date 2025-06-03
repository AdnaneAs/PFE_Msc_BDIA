import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Simple image processing utilities"""
    
    @staticmethod
    def get_image_info(image_path: str) -> Dict[str, Any]:
        """Get basic image information"""
        try:
            with Image.open(image_path) as img:
                return {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                    "size_bytes": os.path.getsize(image_path)
                }
        except Exception as e:
            logger.error(f"Error getting image info for {image_path}: {str(e)}")
            return {}
    
    @staticmethod
    def generate_image_hash(image_path: str) -> str:
        """Generate unique hash for image content"""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating hash for {image_path}: {str(e)}")
            return ""
    
    @staticmethod
    def is_valid_image(image_path: str) -> bool:
        """Check if file is a valid image"""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    @staticmethod
    def resize_image_if_needed(image_path: str, max_size: int = 1024) -> str:
        """Resize image if it's too large (for API efficiency)"""
        try:
            with Image.open(image_path) as img:
                if max(img.width, img.height) > max_size:
                    # Calculate new size maintaining aspect ratio
                    ratio = max_size / max(img.width, img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    
                    # Create resized image
                    resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Save resized version
                    resized_path = image_path.replace('.', '_resized.')
                    resized_img.save(resized_path, img.format)
                    
                    logger.info(f"Resized image {image_path} to {new_size}")
                    return resized_path
                else:
                    return image_path
        except Exception as e:
            logger.error(f"Error resizing image {image_path}: {str(e)}")
            return image_path

def process_document_images(doc_id: str, image_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Process all images from a document for VLM description
    
    Args:
        doc_id: Document ID
        image_paths: List of image file paths
        
    Returns:
        List of processed image metadata
    """
    processed_images = []
    
    for i, image_path in enumerate(image_paths):
        try:
            if not ImageProcessor.is_valid_image(image_path):
                logger.warning(f"Skipping invalid image: {image_path}")
                continue
            
            # Get image information
            image_info = ImageProcessor.get_image_info(image_path)
            image_hash = ImageProcessor.generate_image_hash(image_path)
            
            # Prepare metadata
            image_metadata = {
                "doc_id": doc_id,
                "image_path": image_path,
                "image_index": i,
                "image_hash": image_hash,
                "filename": os.path.basename(image_path),
                "processed_at": datetime.now().isoformat(),
                **image_info
            }
            
            processed_images.append(image_metadata)
            logger.info(f"Processed image metadata for {image_path}")
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            continue
    
    return processed_images

async def generate_image_descriptions(image_metadatas: List[Dict[str, Any]], vlm_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Generate descriptions for all images using VLM
    
    Args:
        image_metadatas: List of image metadata
        vlm_config: VLM configuration
        
    Returns:
        List of image metadata with descriptions
    """
    from app.services.vlm_service import describe_image
    
    enhanced_metadatas = []
    
    for metadata in image_metadatas:
        try:
            image_path = metadata["image_path"]
            
            # Generate description using VLM
            description, model_info = describe_image(image_path, vlm_config)
            
            # Add description to metadata
            enhanced_metadata = {
                **metadata,
                "description": description,
                "vlm_model": model_info,
                "description_generated_at": datetime.now().isoformat()
            }
            
            enhanced_metadatas.append(enhanced_metadata)
            logger.info(f"Generated description for {image_path}: {description[:100]}...")
            
        except Exception as e:
            logger.error(f"Error generating description for {metadata.get('image_path', 'unknown')}: {str(e)}")
            # Add metadata without description for now
            enhanced_metadatas.append({
                **metadata,
                "description": f"Error generating description: {str(e)}",
                "vlm_model": "error",
                "description_generated_at": datetime.now().isoformat()
            })
    
    return enhanced_metadatas

def prepare_image_chunks_for_vectorization(image_metadatas: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Prepare image descriptions and metadata for vector storage
    
    Args:
        image_metadatas: List of image metadata with descriptions
        
    Returns:
        Tuple of (descriptions_for_embedding, metadata_for_storage)
    """
    descriptions = []
    storage_metadatas = []
    
    for metadata in image_metadatas:
        description = metadata.get("description", "")
        
        # Skip if no valid description
        if not description or description.startswith("Error"):
            continue
        
        # Prepare description for embedding (include context)
        doc_filename = metadata.get("filename", "")
        enhanced_description = f"Image from {doc_filename}: {description}"
        
        # Prepare metadata for storage
        storage_metadata = {
            "type": "image",
            "doc_id": metadata["doc_id"],
            "image_path": metadata["image_path"],
            "image_index": metadata["image_index"],
            "image_hash": metadata["image_hash"],
            "filename": metadata["filename"],
            "description": description,
            "vlm_model": metadata["vlm_model"],
            "width": metadata.get("width", 0),
            "height": metadata.get("height", 0),
            "format": metadata.get("format", ""),
            "size_bytes": metadata.get("size_bytes", 0)
        }
        
        descriptions.append(enhanced_description)
        storage_metadatas.append(storage_metadata)
    
    logger.info(f"Prepared {len(descriptions)} image descriptions for vectorization")
    return descriptions, storage_metadatas

logger.info("Image Processing Service initialized")
