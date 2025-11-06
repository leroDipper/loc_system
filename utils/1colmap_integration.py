import torch
import cv2
import numpy as np
import os
import sqlite3
from pathlib import Path

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


def create_colmap_db(db_path):
    """Create COLMAP database with required tables"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create cameras table
    c.execute('''CREATE TABLE IF NOT EXISTS cameras (
        camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        model INTEGER NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        params BLOB,
        prior_focal_length INTEGER NOT NULL
    )''')
    
    # Create images table
    c.execute('''CREATE TABLE IF NOT EXISTS images (
        image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        name TEXT NOT NULL UNIQUE,
        camera_id INTEGER NOT NULL,
        prior_qw REAL,
        prior_qx REAL,
        prior_qy REAL,
        prior_qz REAL,
        prior_tx REAL,
        prior_ty REAL,
        prior_tz REAL,
        FOREIGN KEY(camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE
    )''')
    
    # Create keypoints table
    c.execute('''CREATE TABLE IF NOT EXISTS keypoints (
        image_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL,
        cols INTEGER NOT NULL,
        data BLOB NOT NULL,
        FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
    )''')
    
    # Create descriptors table
    c.execute('''CREATE TABLE IF NOT EXISTS descriptors (
        image_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL,
        cols INTEGER NOT NULL,
        data BLOB NOT NULL,
        FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
    )''')
    
    # Create matches table (empty, will be filled by COLMAP matcher)
    c.execute('''CREATE TABLE IF NOT EXISTS matches (
        pair_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL,
        cols INTEGER NOT NULL,
        data BLOB
    )''')
    
    # Create two_view_geometries table (empty, will be filled by COLMAP)
    c.execute('''CREATE TABLE IF NOT EXISTS two_view_geometries (
        pair_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL,
        cols INTEGER NOT NULL,
        data BLOB,
        config INTEGER NOT NULL,
        F BLOB,
        E BLOB,
        H BLOB,
        qvec BLOB,
        tvec BLOB
    )''')
    
    conn.commit()
    conn.close()


def add_camera_to_db(db_path, camera_id, width, height, fx, fy, cx, cy):
    """
    Add PINHOLE camera model to database
    PINHOLE model (model=1 in COLMAP) has params: fx, fy, cx, cy
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # PINHOLE model = 1
    model = 1
    params = np.array([fx, fy, cx, cy], dtype=np.float64)
    params_blob = params.tobytes()
    
    c.execute('INSERT OR REPLACE INTO cameras VALUES (?, ?, ?, ?, ?, ?)',
              (camera_id, model, width, height, params_blob, 1))
    
    conn.commit()
    conn.close()
    print(f"Added camera {camera_id}: PINHOLE {width}x{height}, fx={fx}, fy={fy}, cx={cx}, cy={cy}")


def add_image_to_db(db_path, image_id, image_name, camera_id, keypoints, descriptors):
    """
    Add image and its features to the database
    
    Args:
        db_path: Path to database
        image_id: Integer ID for this image
        image_name: Name of the image file
        camera_id: Camera ID
        keypoints: (N, 6) COLMAP format keypoints
        descriptors: (N, 64) uint8 descriptors
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Insert image
    c.execute('INSERT OR REPLACE INTO images (image_id, name, camera_id) VALUES (?, ?, ?)',
              (image_id, image_name, camera_id))
    
    # Insert keypoints
    keypoints_blob = keypoints.astype(np.float32).tobytes()
    c.execute('INSERT OR REPLACE INTO keypoints VALUES (?, ?, ?, ?)',
              (image_id, keypoints.shape[0], keypoints.shape[1], keypoints_blob))
    
    # Insert descriptors
    descriptors_blob = descriptors.tobytes()
    c.execute('INSERT OR REPLACE INTO descriptors VALUES (?, ?, ?, ?)',
              (image_id, descriptors.shape[0], descriptors.shape[1], descriptors_blob))
    
    conn.commit()
    conn.close()


def extract_and_populate_database(dataset_path, db_path, camera_params):
    """
    Extract XFeat features and populate COLMAP database
    
    Args:
        dataset_path: Path to directory containing images
        db_path: Path to output COLMAP database
        camera_params: dict with keys: width, height, fx, fy, cx, cy
    """
    # Initialize XFeat
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
    for f in os.listdir(dataset_path):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(dataset_path, f))
    image_paths.sort()
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("No images found!")
        return
    
    # Create database
    print(f"\nCreating database at {db_path}")
    if os.path.exists(db_path):
        os.remove(db_path)
    create_colmap_db(db_path)
    
    # Add camera
    camera_id = 1
    add_camera_to_db(
        db_path, 
        camera_id,
        camera_params['width'],
        camera_params['height'],
        camera_params['fx'],
        camera_params['fy'],
        camera_params['cx'],
        camera_params['cy']
    )
    
    # Process each image
    print("\nExtracting features...")
    for idx, image_path in enumerate(image_paths):
        image_id = idx + 1
        image_name = os.path.basename(image_path)
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read {image_path}")
            continue
            
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extract features
        with torch.no_grad():
            output = xfeat.detectAndCompute(img_gray, top_k=4000)
        
        features = output[0]
        keypoints = features['keypoints'].cpu().numpy()
        descriptors = features['descriptors'].cpu().numpy()
        
        # Convert to COLMAP format
        colmap_kpts, colmap_desc = xfeat_to_colmap_format(keypoints, descriptors)
        
        # Add to database
        add_image_to_db(db_path, image_id, image_name, camera_id, colmap_kpts, colmap_desc)
        
        print(f"  [{image_id}/{len(image_paths)}] {image_name}: {len(keypoints)} keypoints, "
              f"{descriptors.shape[1]}D descriptors")
    
    print(f"\n✓ Database created successfully!")
    print(f"  Path: {db_path}")
    print(f"  Images: {len(image_paths)}")
    print(f"  Camera: PINHOLE {camera_params['width']}x{camera_params['height']}")
    print(f"\nNext steps:")
    print(f"  1. Run COLMAP feature matcher:")
    print(f"     colmap exhaustive_matcher --database_path {db_path}")
    print(f"  2. Run COLMAP reconstruction")


if __name__ == "__main__":
    # Configuration
    dataset_path = 'colmap_database/large_map/large_set_train_640x480'
    output_dir = 'colmap_database/large_map_xfeat'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, 'database.db')
    
    # Camera parameters from your image
    camera_params = {
        'width': 640,
        'height': 480,
        'fx': 768.0,
        'fy': 768.0,
        'cx': 320.0,
        'cy': 240.0
    }
    
    # Run extraction and database creation
    extract_and_populate_database(dataset_path, db_path, camera_params)