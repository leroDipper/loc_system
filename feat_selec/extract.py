import torch
import cv2
import numpy as np
import os
from pathlib import Path
import yaml
import pandas as pd


def load_camera_params_from_yaml(yaml_path):
    """
    Load camera parameters from YAML file
    
    Returns:
        camera_params: np.array([width, height, fx, fy, cx, cy])
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    width, height = data['resolution']
    fx, fy, cx, cy = data['intrinsics']
    
    return np.array([width, height, fx, fy, cx, cy], dtype=np.float64)



def xfeat_to_colmap_format(keypoints, descriptors, scores=None):
    """
    Convert XFeat features to COLMAP format
    
    Args:
        keypoints: (N, 2) array of (x, y) positions
        descriptors: (N, 64) float32 array, L2-normalized, range ~[-0.5, 0.5]
        scores: (N,) array of keypoint scores (optional)
    
    Returns:
        colmap_keypoints: (N, 6) array - x, y, a11, a12, a21, a22
        colmap_descriptors: (N, 64) uint8 array
    """
    N = keypoints.shape[0]
    
    # Create COLMAP keypoints with identity affine shape
    colmap_keypoints = np.zeros((N, 6), dtype=np.float32)
    colmap_keypoints[:, 0] = keypoints[:, 0]  # x
    colmap_keypoints[:, 1] = keypoints[:, 1]  # y
    colmap_keypoints[:, 2] = 1.0  # a11 (identity)
    colmap_keypoints[:, 3] = 0.0  # a12
    colmap_keypoints[:, 4] = 0.0  # a21
    colmap_keypoints[:, 5] = 1.0  # a22 (identity)
    
    # Convert descriptors from [-0.5, 0.5] float to [0, 255] uint8
    colmap_descriptors = np.clip((descriptors + 0.5) * 255.0, 0, 255).astype(np.uint8)
    
    return colmap_keypoints, colmap_descriptors






def collect_feature_rows(image_id, image_name, keypoints, descriptors, scores):
    """
    create a pandas DataFrame for keypoints and descriptors and image id
    """
    rows = []
    n_keypoints = keypoints.shape[0]
    for i in range(n_keypoints):
        rows.append({
            "image_id": image_id,
            "image_name": image_name,
            "keypoint": keypoints[i],
            "descriptor": descriptors[i],
            "detector_scores": scores[i]
        })

    return rows
        
    
    

def extract_and_populate_database(dataset_path, camera_params):
    """
    Extract XFeat features and populate dataframe
    
    
    """

    print("Loading XFeat model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)
        xfeat = xfeat.to(device)
        xfeat.eval()
        print(f"XFeat loaded on {device}")
    except Exception as e:
        print(f"Error loading XFeat model: {e}")
        return
    
    # Get image paths
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_paths.extend(Path(dataset_path).glob(ext))
    image_paths = sorted(image_paths)
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("No images found!")
        return
    
    
    
    # Add camera
    camera_id = 1
    width, height, fx, fy, cx, cy = camera_params

    all_rows = []
   
    # Process each image
    print("\nExtracting features...")
    for idx, image_path in enumerate(image_paths):
        image_id = idx + 1
        image_name = image_path.name
        
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not read {image_path}")
            continue
        
        # Check image size matches camera params
        if img.shape[1] != int(width) or img.shape[0] != int(height):
            print(f"Warning: {image_name} size mismatch! Expected {int(width)}x{int(height)}, got {img.shape[1]}x{img.shape[0]}")
            continue
            
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extract features
        with torch.no_grad():
            output = xfeat.detectAndCompute(img_gray, top_k=2000)
        
        features = output[0]
        keypoints = features['keypoints'].cpu().numpy()
        descriptors = features['descriptors'].cpu().numpy().astype(np.float32)

        
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        descriptors = descriptors / (norms + 1e-8)

        scores = features['scores'].cpu().numpy() 

        rows = collect_feature_rows(image_id, image_name, keypoints, descriptors, scores)
        all_rows.extend(rows)
        
        
        print(f"  [{image_id}/{len(image_paths)}] {image_name}: {len(keypoints)} keypoints")

    df = pd.DataFrame(all_rows)

    return df
    


def save(npz_path, dataframe):
    """
    Save dataframe to npz file
    """
    print(f"\nSaving features to {npz_path}...")
    image_ids = dataframe['image_id'].to_numpy()
    image_names = dataframe['image_name'].to_numpy()
    keypoints = np.stack(dataframe['keypoint'].to_numpy())
    descriptors = np.stack(dataframe['descriptor'].to_numpy())
    detector_scores = dataframe['detector_scores'].to_numpy()
    
    np.savez_compressed(npz_path,
                        image_ids=image_ids,
                        image_names=image_names,
                        keypoints=keypoints,
                        descriptors=descriptors,
                        detector_scores=detector_scores)
    
    print("Save complete.")




if __name__ == "__main__":
    # Configuration
    dataset_path = '/home/leroy-marewangepo/Masters_Stuff/loc_code_test_pi/resources/tum_fr1/images'
    output_dir = '/home/leroy-marewangepo/Masters_Stuff/loc_code_test_pi/resources/tum_fr1/raw_feat'
    yaml_path = '/resources/tum_fr1'
    
    
    # Load camera parameters from YAML (rectified images)
    yaml_path = 'resources/tum_fr1/camera_params.yaml'
    
    if not os.path.exists(yaml_path):
        print(f"ERROR: Camera parameter file not found: {yaml_path}")
        print("Please run rectify_and_resize.py first!")
        exit(1)
    
    camera_params = load_camera_params_from_yaml(yaml_path)
    
   
    print(f"Camera parameters from: {yaml_path}")
    print(f"  Resolution: {int(camera_params[0])}x{int(camera_params[1])}")
    print(f"  Intrinsics: fx={camera_params[2]:.2f}, fy={camera_params[3]:.2f}, cx={camera_params[4]:.2f}, cy={camera_params[5]:.2f}")
    print()
    
    # Run extraction and database creation
    dataframe = extract_and_populate_database(dataset_path, camera_params)

    save_path = os.path.join(output_dir, 'tum_fr1_features_xfeat.npz')
    save(save_path, dataframe)


