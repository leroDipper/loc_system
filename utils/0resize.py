import cv2
from pathlib import Path

def resize_images(input_dir, output_dir, target_width=640, target_height=480):
    """
    Resize all images in a directory.
    
    Args:
        input_dir: Path to input images (e.g., 'improved_office_dataset/left')
        output_dir: Path to save resized images (e.g., 'improved_office_dataset/left_640x480')
        target_width: New width (default: 640)
        target_height: New height (default: 480)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all jpg images
    image_files = sorted(input_path.glob("*.jpg"))
    
    print(f"Resizing {len(image_files)} images from {input_dir}")
    print(f"Target size: {target_width}×{target_height}")
    print(f"Output: {output_dir}")
    
    for i, img_path in enumerate(image_files):
        # Read image
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"  ⚠ Could not read {img_path.name}")
            continue
        
        # Resize
        resized = cv2.resize(img, (target_width, target_height))
        
        # Save with same filename
        output_file = output_path / img_path.name
        cv2.imwrite(str(output_file), resized)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(image_files)}...")
    
    print(f"✓ Done! Saved {len(image_files)} resized images to {output_dir}")


if __name__ == "__main__":
    # Resize your training images
    resize_images(
        input_dir='colmap_database/large_map/large_set_test',
        output_dir='colmap_database/large_map/large_set_train_test_640x480',
        target_width=640,
        target_height=480
    )