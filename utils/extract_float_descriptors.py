#!/usr/bin/env python3
"""
Simple script to extract float32 L2-normalized descriptors from images.
Does NOT touch COLMAP database - just creates float_descriptors.npz
"""

import torch
import cv2
import numpy as np
from pathlib import Path


def extract_float_descriptors(dataset_path, output_path):
    """
    Extract XFeat descriptors and save as float32 L2-normalized.
    
    Args:
        dataset_path: Path to directory containing images
        output_path: Path to save float_descriptors.npz
    """
    # Initialize XFeat
    print("Loading XFeat model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)
    xfeat = xfeat.to(device)
    xfeat.eval()
    print(f"XFeat loaded on {device}")
    
    # Get image paths
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_paths.extend(Path(dataset_path).glob(ext))
    image_paths = sorted(image_paths)
    print(f"Found {len(image_paths)} images")
    
    # Storage for float descriptors
    float_descriptors_data = {}
    
    # Process each image
    print("\nExtracting features...")
    for idx, image_path in enumerate(image_paths):
        image_name = image_path.name
        
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not read {image_path}")
            continue
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extract features
        with torch.no_grad():
            output = xfeat.detectAndCompute(img_gray, top_k=2000)
        
        features = output[0]
        descriptors = features['descriptors'].cpu().numpy()
        
        # L2 normalize
        desc_float = descriptors.astype(np.float32)
        desc_float = desc_float / (np.linalg.norm(desc_float, axis=1, keepdims=True) + 1e-8)
        
        # Store
        float_descriptors_data[image_name] = desc_float
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(image_paths)} images")
    
    # Save
    np.savez_compressed(output_path, **float_descriptors_data)
    print(f"\n✓ Saved float32 descriptors to {output_path}")
    print(f"  Total images: {len(float_descriptors_data)}")
    print(f"  Descriptor shape example: {float_descriptors_data[list(float_descriptors_data.keys())[0]].shape}")


if __name__ == "__main__":
    # Configuration for FR1
    dataset_path = 'resources/tum_fr3/images'
    output_path = 'resources/tum_fr3/float_descriptors.npz'
    
    print("="*60)
    print("FLOAT DESCRIPTOR EXTRACTION")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_path}")
    print()
    
    extract_float_descriptors(dataset_path, output_path)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
