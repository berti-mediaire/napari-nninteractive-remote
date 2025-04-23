from flask import Flask, request, jsonify
import torch
import io
import numpy as np
import os
from huggingface_hub import snapshot_download
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
import SimpleITK as sitk  # Import SimpleITK for potential image handling
import json # Import json for handling spacing

app = Flask(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Model Loading from Hugging Face Hub ---
REPO_ID = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"
DOWNLOAD_DIR = "download_models_server"  # Different download dir for server to avoid conflicts

# Ensure download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

download_path = snapshot_download(
    repo_id=REPO_ID,
    allow_patterns=[f"{MODEL_NAME}/*"],
    local_dir=DOWNLOAD_DIR,
)

model_path = os.path.join(DOWNLOAD_DIR, MODEL_NAME)

# Initialize the Inference Session (we'll do this once at server start)
try:
    session = nnInteractiveInferenceSession(
        device=device,
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=os.cpu_count(),
        do_autozoom=False,  # Server doesn't need autozoom
        use_pinned_memory=True,
    )
    session.initialize_from_trained_model_folder(model_path)
    print("nnInteractiveInferenceSession initialized successfully on:", device)
except Exception as e:
    print(f"Error initializing inference session: {e}")
    session = None

def prepare_image(image_bytes, shape_str, dtype_str, spacing_str):
    try:
        shape = tuple(map(int, json.loads(shape_str)))
        dtype = np.dtype(dtype_str)
        spacing = json.loads(spacing_str) if spacing_str != 'None' else None
        image_array = np.frombuffer(image_bytes, dtype=dtype).reshape(shape)
        image_array = image_array.astype(np.float32) # Ensure correct type
        return image_array, spacing
    except Exception as e:
        return f"Error loading/preparing image: {e}", None

def numpy_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError (f"Type {type(obj)} is not JSON serializable")

@app.route('/segment', methods=['POST'])
def segment_image():
    if session is None:
        return jsonify({'error': 'Inference session not initialized'}), 503

    if 'Content-Type' not in request.headers or request.headers['Content-Type'] != 'application/octet-stream':
        return jsonify({'error': 'Expected application/octet-stream for image data'}), 400

    image_bytes = request.data
    shape_str = request.headers.get('Image-Shape')
    dtype_str = request.headers.get('Image-Dtype')
    spacing_str = request.headers.get('Spacing')

    if not all([image_bytes, shape_str, dtype_str]):
        return jsonify({'error': 'Missing image data or metadata in headers'}), 400

    image_array, spacing = prepare_image(image_bytes, shape_str, dtype_str, spacing_str)

    if isinstance(image_array, str):
        return jsonify({'error': image_array}), 400

    try:
        session.set_image(image_array, {"spacing": spacing})
        target_tensor = torch.zeros(image_array.shape[1:], dtype=torch.uint8).to(device)
        session.set_target_buffer(target_tensor)
        results = session.target_buffer.cpu().numpy()
        return jsonify({'segmentation': results.tolist()})
    except Exception as e:
        return jsonify({'error': f"Error during initial segmentation: {e}"}), 500

@app.route('/reset_interaction', methods=['POST'])
def reset_interaction():
    if session:
        session.reset_interactions()
        return jsonify({'status': 'Interaction state reset'})
    return jsonify({'error': 'Inference session not initialized'}), 503

@app.route('/interact', methods=['POST'])
def interact():
    # ... (rest of your /interact function - it seems to be handling JSON correctly) ...
    pass

@app.route('/init_mask', methods=['POST'])
def init_mask():
    # ... (rest of your /init_mask function - it seems to be handling JSON correctly) ...
    pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)