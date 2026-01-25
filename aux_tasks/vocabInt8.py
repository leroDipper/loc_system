"""
rebuild_vocab_int8_mh01.py

Rebuild vocabulary tree using INT8 ONNX model descriptors
to match the quantized descriptor distribution used during query time.
"""

import onnxruntime as ort
import os
import cv2
import numpy as np
import time
from vocabTree import VocabTreeBuilder, VocabTreeNode

def load_colmap_image_names(images_txt_path):
    """Load image names from COLMAP images.txt in order."""
    colmap_images = []
    with open(images_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) == 10:
                colmap_images.append(parts[9])
    return colmap_images

if __name__ == "__main__":
    # Configuration
    EUROC_DATASET_PATH = 'resources/mh_01'
    IMAGE_DIR = os.path.join(EUROC_DATASET_PATH, 'images')
    N_TRAIN_IMAGES = 1500
    TOP_K_FEATURES = 2000  # Extract more features for better vocabulary coverage
    
    # Load INT8 ONNX model
    print("Loading INT8 ONNX model...")
    int8_model_path = 'models/xfeat_752x480.onnx'
    session = ort.InferenceSession(int8_model_path, providers=['CPUExecutionProvider'])
    print("✓ INT8 model loaded")
    
    # Get training images (same split as used for map building)
    colmap_images = load_colmap_image_names('resources/mh_01/proj_files/images.txt')
    train_images = colmap_images[:N_TRAIN_IMAGES]
    
    print(f"\nExtracting INT8 descriptors from {len(train_images)} training images...")
    print("(This will take a few minutes)")
    
    int8_descriptors = []
    
    for i, frame_name in enumerate(train_images):
        frame_path = os.path.join(IMAGE_DIR, frame_name)
        
        # Read and process frame
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not load {frame_name}, skipping...")
            continue
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Preprocess for ONNX (same as query pipeline)
        frame_input = frame_gray.astype(np.float32)
        frame_input = np.expand_dims(frame_input, axis=0)
        frame_input = np.expand_dims(frame_input, axis=0)
        
        # Extract features using INT8 ONNX
        feats, keypoints_logits, heatmap = session.run(None, {'input': frame_input})
        
        # Process outputs to get sparse features (same as query pipeline)
        B, C, H, W = feats.shape
        
        # Get heatmap scores and select top-k (extract many features for vocabulary)
        heat_flat = heatmap[0, 0].flatten()
        
        if len(heat_flat) > TOP_K_FEATURES:
            top_indices = np.argpartition(heat_flat, -TOP_K_FEATURES)[-TOP_K_FEATURES:]
        else:
            top_indices = np.arange(len(heat_flat))
        
        # Get descriptors
        feats_flat = feats[0].reshape(64, -1).T
        descriptors = feats_flat[top_indices]
        
        # L2 normalize (same as query pipeline)
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        descriptors = descriptors / (norms + 1e-8)
        
        # Convert to uint8 (same as query pipeline)
        descriptors = np.clip((descriptors + 0.5) * 255.0, 0, 255).astype(np.uint8)
        
        int8_descriptors.append(descriptors)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(train_images)} images...")
    
    # Stack all descriptors
    int8_descriptors = np.vstack(int8_descriptors)
    print(f"\n✓ Extracted {len(int8_descriptors)} total descriptors")
    print(f"  Descriptor shape: {int8_descriptors.shape}")
    print(f"  Descriptor dtype: {int8_descriptors.dtype}")
    print(f"  Descriptor range: [{int8_descriptors.min()}, {int8_descriptors.max()}]")
    
    # Build vocabulary tree with INT8 descriptors
    print("\nBuilding vocabulary tree...")
    n_branches = 10
    depth = 4
    descriptor_dim = int8_descriptors.shape[1]
    
    t_start = time.time()
    builder = VocabTreeBuilder(n_branches, depth, descriptor_dim)
    
    # VocabTreeBuilder expects float32 input
    builder.build(int8_descriptors.astype(np.float32))
    
    t_build = time.time() - t_start
    print(f"✓ Vocabulary tree built in {t_build:.2f}s")
    
    # Save vocabulary
    output_path = 'resources/mh_01/vocabularies/vocab_tree_int8.bin'
    builder.save(output_path)
    print(f"✓ Saved INT8-compatible vocabulary to: {output_path}")
    
    # Verify vocabulary
    vocab = builder.get_vocabulary()
    print(f"\nVocabulary stats:")
    print(f"  Shape: {vocab.shape}")
    print(f"  Expected leaf nodes: {n_branches ** depth} = {10**4}")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print("Now update mh_01_onnx.py line 62 to use:")
    print(f"  vocabulary = '{output_path}'")
    print("="*60)