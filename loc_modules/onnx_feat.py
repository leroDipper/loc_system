import onnxruntime as ort
import numpy as np
import time
from test.memory import MemoryMonitor


def onnx_extractor(session, img_gray, top_k):
    
    #MemoryMonitor.print_memory("After loading ONNX model")

    t_start = time.time()
    frame_input = img_gray.astype(np.float32)
    frame_input = np.expand_dims(frame_input, axis=0)
    frame_input = np.expand_dims(frame_input, axis=0)
   
    # Extract features using INT8 ONNX
    feats, keypoints_logits, heatmap = session.run(None, {'input': frame_input})
    # Extract features using INT8 ONNX
    feats, keypoints_logits, heatmap = session.run(None, {'input': frame_input})
    t_extract = time.time() - t_start

    # Process outputs to get sparse features 
    B, C, H, W = feats.shape

    # Get heatmap scores and select top-k
    heat_flat = heatmap[0, 0].flatten()

    if len(heat_flat) > top_k:
        top_indices = np.argpartition(heat_flat, -top_k)[-top_k:]
    else:
        top_indices = np.arange(len(heat_flat))
    
    # Get keypoint positions
    y_coords = np.repeat(np.arange(H), W)
    x_coords = np.tile(np.arange(W), H)
    kpts_x = x_coords[top_indices] * 8
    kpts_y = y_coords[top_indices] * 8
    keypoints = np.stack([kpts_x, kpts_y], axis=1).astype(np.float32)

     # Get descriptors
    feats_flat = feats[0].reshape(64, -1).T
    descriptors = feats_flat[top_indices]
    
    # L2 normalize
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors = descriptors / (norms + 1e-8)

    return keypoints, descriptors, t_extract
    






