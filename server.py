from flask import Flask, request, jsonify
import torch
import io
import numpy as np
import os
from huggingface_hub import snapshot_download
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
import SimpleITK as sitk  # Import SimpleITK for potential image handling

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

def prepare_image(image_bytes):
    try:
        sitk_image = sitk.ReadImage(io.BytesIO(image_bytes))
        image_array = sitk.GetArrayFromImage(sitk_image)
        image_array = np.expand_dims(image_array, axis=0).astype(np.float32) # Ensure correct shape and type
        return image_array, sitk_image.GetSpacing()
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

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    image_array, spacing = prepare_image(image_bytes)

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
    if session is None:
        return jsonify({'error': 'Inference session not initialized'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No interaction data provided'}), 400

    interaction_type = data.get('type')
    interaction_data = data.get('data')
    prompt = data.get('prompt')
    auto_run = data.get('auto_run')
    spacing = data.get('spacing') # You might need to send spacing with each interaction

    if not all([interaction_type, interaction_data]):
        return jsonify({'error': 'Missing interaction parameters'}), 400

    interaction_array = np.array(interaction_data)

    try:
        if interaction_type == "point":
            session.add_point_interaction(interaction_array, prompt, auto_run)
        elif interaction_type == "bbox":
            session.add_bbox_interaction(interaction_array, prompt, auto_run)
        elif interaction_type == "scribble":
            session.add_scribble_interaction(interaction_array.astype(np.uint8), prompt, auto_run)
        elif interaction_type == "lasso":
            session.add_lasso_interaction(interaction_array, prompt, auto_run)
        else:
            return jsonify({'error': f'Unknown interaction type: {interaction_type}'}), 400

        results = session.target_buffer.cpu().numpy()
        return jsonify({'segmentation': results.tolist()})
    except Exception as e:
        return jsonify({'error': f"Error during interaction: {e}"}), 500

@app.route('/init_mask', methods=['POST'])
def init_mask():
    if session is None:
        return jsonify({'error': 'Inference session not initialized'}), 503

    data = request.get_json()
    if not data or 'initial_mask' not in data:
        return jsonify({'error': 'No initial mask provided'}), 400

    initial_mask = np.array(data['initial_mask'], dtype=np.uint8)
    run_prediction = data.get('run_prediction', False)
    spacing = data.get('spacing') # You might need to send spacing

    try:
        session.add_initial_seg_interaction(initial_mask, run_prediction=run_prediction)
        results = session.target_buffer.cpu().numpy()
        return jsonify({'segmentation': results.tolist()})
    except Exception as e:
        return jsonify({'error': f"Error during initial mask interaction: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)