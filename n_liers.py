import glob
import os

all_frames = sorted(glob.glob('resources/tum_fr2/images/*.png'))
all_filenames = [os.path.basename(f) for f in all_frames]

# Load colmap set
colmap_images = []
with open('resources/tum_fr2/project_files/images.txt', 'r') as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) == 10:
            colmap_images.append(parts[9])
colmap_set = set(colmap_images)

reconstructed_chrono = [f for f in all_filenames if f in colmap_set]

train = reconstructed_chrono[:2200]
test = reconstructed_chrono[2200:]

print(f"Last train frame:  {train[-1]}")
print(f"First test frame:  {test[0]}")
print(f"Test frame 60:     {test[59]}")
print(f"Test frame 360:    {test[359]}")
print(f"Last test frame:   {test[-1]}")