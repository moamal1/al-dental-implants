"""
UNet model loading and inference for IAN nerve segmentation.
"""

import numpy as np
import torch
from scipy.ndimage import zoom
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

import config


def load_model(model_path=None, device=None):
    """Load the pre-trained UNet model."""
    if model_path is None:
        model_path = config.MODEL_PATH
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=config.UNET_CHANNELS,
        strides=config.UNET_STRIDES,
        num_res_units=config.UNET_NUM_RES_UNITS,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, device


def preprocess_image(image, target_max_dim=None):
    """Normalize and optionally downsample image for inference.

    Returns:
        (processed_image_tensor, downsampled_shape)
    """
    if target_max_dim is None:
        target_max_dim = config.TARGET_MAX_DIM

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    orig_shape = image.shape

    max_dim = max(orig_shape)
    scale = min(1.0, target_max_dim / max_dim)

    if scale < 1.0:
        small_img = zoom(image, zoom=(scale, scale, scale), order=1)
    else:
        small_img = image.copy()

    small_img = np.nan_to_num(small_img, nan=0.0, posinf=0.0, neginf=0.0)
    min_val, max_val = small_img.min(), small_img.max()

    if float(max_val - min_val) > 1e-8:
        small_img = (small_img - min_val) / (max_val - min_val)
    else:
        small_img = np.zeros_like(small_img, dtype=np.float32)

    tensor = torch.from_numpy(small_img).unsqueeze(0).unsqueeze(0).float()
    return tensor, small_img.shape


def run_inference(model, inputs, device):
    """Run sliding-window inference and return the predicted mask (small scale)."""
    post_pred = AsDiscrete(argmax=True)

    with torch.no_grad():
        outputs = sliding_window_inference(
            inputs=inputs.to(device),
            roi_size=config.ROI_SIZE,
            sw_batch_size=config.SW_BATCH_SIZE,
            predictor=model,
        )
        outputs_list = [post_pred(i) for i in decollate_batch(outputs)]
        pred_small = outputs_list[0].cpu().numpy()

    return np.squeeze(pred_small).astype(np.uint8)


def postprocess_prediction(pred_small, orig_shape):
    """Upsample the predicted mask back to the original image shape."""
    if pred_small.shape != orig_shape:
        zoom_factors = tuple(
            orig_shape[i] / pred_small.shape[i] for i in range(3)
        )
        pred_mask = zoom(pred_small.astype(np.float32), zoom=zoom_factors, order=0)
    else:
        pred_mask = pred_small

    return (pred_mask > 0.5).astype(np.uint8)
