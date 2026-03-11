import sqlite3
import numpy as np
import os

# Connect to database
conn = sqlite3.connect('resources/iphone/iphone.db')
cursor = conn.cursor()

# Create output directory
os.makedirs('resources/iphone/project_files_text/descriptors', exist_ok=True)

# Get all images
cursor.execute("SELECT image_id, name FROM images")
images = cursor.fetchall()

print(f"Found {len(images)} images in database")

for img_id, img_name in images:
    # Get descriptors for this image
    cursor.execute("SELECT data, rows, cols FROM descriptors WHERE image_id = ?", (img_id,))
    result = cursor.fetchone()
    
    if result:
        desc_blob, rows, cols = result
        # XFeat descriptors are 64-dimensional, uint8
        descriptors = np.frombuffer(desc_blob, dtype=np.uint8).reshape(rows, cols)
        
        # Save as text file
        output_file = f'resources/iphone/project_files_text/descriptors/{img_name}_desc.txt'
        np.savetxt(output_file, descriptors, fmt='%d', delimiter=' ')
        print(f"Image: {img_name} - {descriptors.shape[0]} descriptors ({descriptors.shape[1]}D) saved to {output_file}")
    else:
        print(f"No descriptors found for {img_name}")

conn.close()
print("\nDone! Descriptors saved in descriptors/ folder")