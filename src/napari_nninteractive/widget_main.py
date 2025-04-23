import os
import warnings
from pathlib import Path
from typing import Any, Optional

import nnInteractive
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from napari.utils.notifications import show_warning
from napari.viewer import Viewer
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from qtpy.QtWidgets import QWidget
import requests
import io
from PIL import Image
from skimage.transform import resize

from napari_nninteractive.widget_controls import LayerControls

# --- Configuration for the remote server ---
REMOTE_SERVER_URL = "http://192.168.8.181:5000/segment"  # Replace with server URL for a different GPU

class nnInteractiveWidget_(LayerControls):
    """Just a Debug Dummy without all the machine learning stuff"""


class nnInteractiveWidget(LayerControls):
    """
    A widget for the nnInteractive plugin in Napari that manages model inference sessions
    and allows interactive layer-based actions, with remote inference.
    """

    def __init__(self, viewer: Viewer, parent: Optional[QWidget] = None):
        """
        Initialize the nnInteractiveWidget.
        """
        super().__init__(viewer, parent)
        self.session = None
        self._viewer.dims.events.order.connect(self.on_axis_change)
        self.use_remote_inference = True  # Add a flag to control local/remote

    # Event Handlers
    def on_init(self, *args, **kwargs):
        """
        Initialize the inference process. If remote inference is enabled,
        it sends the initial image to the remote server. Otherwise, it initializes locally.
        """
        super().on_init(*args, **kwargs)



        _data = np.array(self._viewer.layers[self.session_cfg["name"]].data)
        _data = _data[np.newaxis, ...]
        if self.source_cfg["ndim"] == 2:
            _data = _data[np.newaxis, ...]
        self._current_image_data = _data  # Store for sending to server

        if self.use_remote_inference:
            self._send_image_to_server()
            # We won't initialize the local session in this case
        else:
            self._initialize_local_session()

        self._data_result = np.zeros(self.session_cfg["shape"], dtype=np.uint8) # Initialize result buffer
        self._viewer.add_image(self._data_result, name=self.label_layer_name, visible=True)

        self._scribble_brush_size = 5 # Default scribble size for remote mode
        if self.scribble_layer_name in self._viewer.layers:
            self._viewer.layers[self.scribble_layer_name].brush_size = self._scribble_brush_size

        # Set the prompt type to positive
        self.prompt_button._uncheck()
        self.prompt_button._check(0)

    def _initialize_local_session(self):
        """Initializes the nnInteractiveInferenceSession locally."""
        if self.session is None:
            # Get inference class from Checkpoint
            if Path(self.checkpoint_path).joinpath("inference_session_class.json").is_file():
                inference_class = load_json(
                    Path(self.checkpoint_path).joinpath("inference_session_class.json")
                )
                if isinstance(inference_class, dict):
                    inference_class = inference_class["inference_class"]
            else:
                inference_class = "nnInteractiveInferenceSession"

            inference_class = recursive_find_python_class(
                join(nnInteractive.__path__[0], "inference"),
                inference_class,
                "nnInteractive.inference",
            )

            # CPU Fallback if no Cuda is available locally
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                show_warning(
                    "Cuda is not available locally. Using CPU instead. This will result in longer runtimes and additionally auto-zoom will be disabled for runtime reasons"
                )
                device = torch.device("cpu")
                self.propagate_ckbx.setChecked(False)

            # Initialize the Session
            self.session = inference_class(
                device=device,  # can also be cpu or mps. CPU not recommended
                use_torch_compile=False,
                torch_n_threads=os.cpu_count(),
                verbose=False,
                do_autozoom=self.propagate_ckbx.isChecked(),
            )

            self.session.initialize_from_trained_model_folder(
                self.checkpoint_path,
                0,
                "checkpoint_final.pth",
            )

        if self.session is not None:
            self.session.set_image(self._current_image_data, {"spacing": self.session_cfg["spacing"]})
            self.session.set_target_buffer(self._data_result)
            self._scribble_brush_size = self.session.preferred_scribble_thickness[
                self._viewer.dims.not_displayed[0]
            ]
            if self.scribble_layer_name in self._viewer.layers:
                self._viewer.layers[self.scribble_layer_name].brush_size = self._scribble_brush_size

    def _send_image_to_server(self):
        """Sends the initial image data to the remote server."""
        if self._current_image_data is not None:
            try:
                img_bytes = self._current_image_data.tobytes()
                headers = {'Content-Type': 'application/octet-stream',
                        'Image-Shape': str(self._current_image_data.shape),
                        'Image-Dtype': str(self._current_image_data.dtype),
                        'Spacing': str(self.session_cfg.get("spacing", None))} # Send spacing as header

                response = requests.post(f"{REMOTE_SERVER_URL}/segment", data=img_bytes, headers=headers)
                response.raise_for_status()
                result_data = response.json()
                print(f"Initial image sent. Server response: {result_data}")
                # You might want to store some initial information from the server here
            except requests.exceptions.RequestException as e:
                show_warning(f"Error sending initial image to server: {e}")
            except Exception as e:
                show_warning(f"Error preparing image for server: {e}")

    def on_model_selected(self):
        """Reset the current session completely"""
        super().on_model_selected()
        self.session = None

    def on_image_selected(self):
        """Reset the current sessions interaction but keep the session itself"""
        super().on_image_selected()
        if self.use_remote_inference:
            self._send_image_to_server()
            # Potentially reset remote inference state
        elif self.session is not None:
            self.session.reset_interactions()

    def on_reset_interactions(self):
        """Reset only the current interaction"""
        _ind = self.interaction_button.index
        super().on_reset_interactions()
        if self.use_remote_inference:
            # Need to tell the server to reset interactions
            try:
                response = requests.post(f"{REMOTE_SERVER_URL}/reset_interaction") # Define this endpoint on the server
                response.raise_for_status()
                print("Remote interactions reset.")
            except requests.exceptions.RequestException as e:
                show_warning(f"Error resetting remote interactions: {e}")
        elif self.session is not None:
            self.session.reset_interactions()

        self._viewer.layers[self.label_layer_name].refresh()
        self.interaction_button._check(_ind)
        self.on_interaction_selected()
        self.prompt_button._on_button_pressed(0)

    def on_next(self):
        """Reset the Interactions of current session"""
        _ind = self.interaction_button.index
        super().on_next()
        if self.use_remote_inference:
            # Need to tell the server to reset interactions for the next step
            try:
                response = requests.post(f"{REMOTE_SERVER_URL}/reset_interaction") # Define this endpoint
                response.raise_for_status()
                print("Remote interactions reset for next step.")
            except requests.exceptions.RequestException as e:
                show_warning(f"Error resetting remote interactions for next: {e}")
        elif self.session is not None:
            self.session.reset_interactions()

        self._viewer.layers[self.label_layer_name].refresh()
        self.interaction_button._check(_ind)
        self.on_interaction_selected()
        self.prompt_button._check(0)

    def on_propagate_ckbx(self, *args, **kwargs):
        if self.session is not None and not self.use_remote_inference:
            self.session.set_do_autozoom(self.propagate_ckbx.isChecked())

    def on_axis_change(self, event: Any):
        """Change the brush size of the scribble layer when the axis changes"""
        if not self.use_remote_inference and self.session is not None:
            self._scribble_brush_size = self.session.preferred_scribble_thickness[
                self._viewer.dims.not_displayed[0]
            ]
            if self.scribble_layer_name in self._viewer.layers:
                self._viewer.layers[self.scribble_layer_name].brush_size = self._scribble_brush_size

    # Inference Behaviour

    def add_interaction(self):
        _index = self.interaction_button.index
        _layer_name = self.layer_dict.get(_index)
        if (
            _layer_name is not None
            and _layer_name in self._viewer.layers
            and not self._viewer.layers[_layer_name].is_free()
        ):
            data = self._viewer.layers[_layer_name].get_last()

            if data is not None:
                _prompt = self.prompt_button.index == 0
                _auto_run = self.run_ckbx.isChecked()

                if self.use_remote_inference:
                    self._send_interaction_to_server(_index, data, _prompt, _auto_run)
                else:
                    self._run_local_interaction(_index, data, _prompt, _auto_run)

    def _run_local_interaction(self, index: int, data: np.ndarray, prompt: bool, auto_run: bool):
        """Runs the interaction using the local inference session."""
        if self.session is None:
            show_warning("Local inference session not initialized.")
            return

        self._viewer.layers[self.layer_dict.get(index)].run()

        if index == 0:
            self._viewer.layers[self.point_layer_name].refresh(force=True)
            self.session.add_point_interaction(data, prompt, auto_run)
        elif index == 1:
            _min = np.min(data, axis=0)
            _max = np.max(data, axis=0)
            bbox = [[_min[0], _max[0]], [_min[1], _max[1]], [_min[2], _max[2]]]
            self.session.add_bbox_interaction(bbox, prompt, auto_run)
        elif index == 2:
            self.session.add_scribble_interaction(data, prompt, auto_run)
        elif index == 3:
            self.session.add_lasso_interaction(data, prompt, auto_run)

        self._viewer.layers[self.label_layer_name].refresh()

    def _send_interaction_to_server(self, index: int, data: np.ndarray, prompt: bool, auto_run: bool):
        """Sends the interaction data to the remote server."""
        try:
            interaction_type = ["point", "bbox", "scribble", "lasso"][index]
            payload = {
                "type": interaction_type,
                "data": data.tolist(),
                "prompt": prompt,
                "auto_run": auto_run,
                "image_shape": self._current_image_data.shape, # Send image context
                "spacing": self.session_cfg["spacing"] # Send spacing context
            }
            response = requests.post(f"{REMOTE_SERVER_URL}/interact", json=payload) # Define /interact endpoint
            response.raise_for_status()
            result_data = response.json()
            self._update_segmentation_from_server(result_data.get("segmentation"))
        except requests.exceptions.RequestException as e:
            show_warning(f"Error sending interaction to server: {e}")
        except Exception as e:
            show_warning(f"Error preparing interaction data: {e}")

    def _update_segmentation_from_server(self, segmentation_data):
        """Updates the local segmentation layer with data from the server."""
        if segmentation_data is not None:
            try:
                segmentation = np.array(segmentation_data, dtype=np.uint8)
                # Resize the segmentation to match the original image shape
                original_shape = self.session_cfg["shape"]
                resized_segmentation = resize(segmentation, original_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
                self._data_result = resized_segmentation
                if self.label_layer_name in self._viewer.layers:
                    self._viewer.layers[self.label_layer_name].data = self._data_result
            except Exception as e:
                show_warning(f"Error updating segmentation from server: {e}")

    def on_load_mask(self):
        if self.use_remote_inference:
            self._send_initial_mask_to_server()
        else:
            self._load_local_mask()

    def _load_local_mask(self):
        _layer_data = self._viewer.layers[self.label_for_init.currentText()].data
        assert _layer_data.shape == self.session_cfg["shape"]
        data = _layer_data == self.class_for_init.value()
        if np.any(data):
            if self.session is not None:
                self.session.add_initial_seg_interaction(
                    data.astype(np.uint8), run_prediction=self.auto_refine.isChecked()
                )
                self._viewer.layers[self.label_layer_name].refresh()
        else:
            warnings.warn("Mask is not valid - probably its empty", UserWarning, stacklevel=1)

    def _send_initial_mask_to_server(self):
        """Sends an initial mask to the remote server."""
        if self.label_for_init.currentText() in self._viewer.layers:
            mask_layer_data = self._viewer.layers[self.label_for_init.currentText()].data
            target_class = self.class_for_init.value()
            initial_mask = (mask_layer_data == target_class).astype(np.uint8)

            try:
                payload = {
                    "initial_mask": initial_mask.tolist(),
                    "run_prediction": self.auto_refine.isChecked(),
                    "image_shape": self._current_image_data.shape, # Send image context
                    "spacing": self.session_cfg["spacing"] # Send spacing context
                }
                response = requests.post(f"{REMOTE_SERVER_URL}/init_mask", json=payload) # Define /init_mask endpoint
                response.raise_for_status()
                result_data = response.json()
                self._update_segmentation_from_server(result_data.get("segmentation"))
            except requests.exceptions.RequestException as e:
                show_warning(f"Error sending initial mask to server: {e}")
            except Exception as e:
                show_warning(f"An unexpected error occurred: {e}")